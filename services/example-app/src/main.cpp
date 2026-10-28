// example-app：多路视频 → 推理 → 跟踪 → Redis 消息 / 分段录像 / RTSP 输出。
// 本文件只做组装：拓扑在 plan.cpp，配置在 config.cpp，其余机制由框架提供。

#include <chrono>
#include <memory>

#include <spdlog/spdlog.h>

#include "config.h"
#include "dsf/app/app.h"
#include "dsf/app/components.h"
#include "dsf/ds/message.h"
#include "dsf/ds/meta_tap.h"
#include "dsf/record/segment_recorder.h"
#include "plan.h"

namespace {

using dsf::publish::Channel;

struct ExampleApp {
  using Config = example_app::Config;

  static dsf::app::Info Describe() {
    return {"example-app", "多路视频 → 推理 → 跟踪 → Redis 消息 / 分段录像 / RTSP 输出",
            [] { return "deepstream=" + dsf::ds::DeepStreamVersion(); }};
  }

  static Config Load(dsf::config::Reader& root) { return example_app::Load(root); }

  static void Build(const Config& config, dsf::app::Assembly& assembly) {
    const example_app::Plan plan = example_app::BuildGraph(config, assembly.graph());
    // RTSP 输出断线（例如 MediaMTX 重启）会报错且不会重连：先排空再退出，录像器封装当前分段，
    // 再由容器重启恢复推流；不排空时每路的当前分段都会丢失。
    assembly.options().drain_on_error = true;
    // 相机可能晚于应用上线（例如断电后同时启动）：不设启动超时，未连上的源与运行中断流一样由
    // 看门狗（watchdog.stall_timeout_s）把守。否则每 30 秒退出重启一次，反复重新加载模型。
    assembly.options().startup_timeout = std::chrono::milliseconds(0);
    for (const auto& warning : example_app::Warnings(config)) spdlog::warn("{}", warning);
    // 断流恢复后的第一帧编成关键帧：录像在断流处干净地切分，RTSP 观看端立即恢复画面。
    if (config.encoder.keyframe_on_gap_ms > 0) {
      const std::chrono::milliseconds gap(config.encoder.keyframe_on_gap_ms);
      for (const auto& encoder : plan.encoders) assembly.Add<dsf::app::KeyframeOnGap>(encoder, gap);
    }

    dsf::ds::SourceRegistry sources;
    for (const auto& source : config.sources) sources.Add(source.id);

    Channel* detections = nullptr;
    Channel* recordings = nullptr;
    if (config.redis.enabled) {
      auto& publisher = assembly.AddPublisher(dsf::publish::MakeRedisStreamPublisher(config.redis.options));
      const int capacity = config.redis.queue_capacity;
      const int max_length = config.redis.max_length;
      const std::size_t bytes = config.redis.queue_bytes;
      detections = &publisher.Open({config.redis.detection_stream, capacity, max_length, bytes});
      recordings = &publisher.Open({config.redis.recording_stream, capacity, max_length, bytes});
    }

    // 每路源的录像器，供检测结果给当前分段打“出现过目标”的标记。
    std::vector<std::vector<dsf::record::SegmentRecorder*>> by_source(config.sources.size());
    for (const auto& slot : plan.recorders) {
      const std::string source_id = config.sources[static_cast<std::size_t>(slot.source)].id;
      const std::string branch = slot.branch;
      const auto publish = [recordings, source_id, branch](dsf::ds::RecordingEvent event) {
        if (recordings == nullptr) return;
        event.source_id = source_id;
        event.branch = branch;
        recordings->TryPublish(dsf::ds::Serialize(dsf::ds::RecordingMessage(event)));
      };
      dsf::record::SegmentRecorder::Events events;
      events.started = [publish](const dsf::record::Segment& segment) {
        dsf::ds::RecordingEvent event;
        event.segment_id = segment.id;
        event.start_ms = segment.start_ms;
        publish(event);
      };
      events.finished = [publish](const dsf::record::Segment& segment, std::uint64_t end_ms) {
        dsf::ds::RecordingEvent event;
        event.type = dsf::ds::RecordingEvent::Type::kFinished;
        event.segment_id = segment.id;
        event.start_ms = segment.start_ms;
        event.end_ms = end_ms;
        event.detected = segment.detected;
        event.path = segment.final_path;
        publish(event);
      };
      dsf::record::RecorderOptions options{config.recording.root, source_id, branch, slot.element};
      options.retention = config.recording.retention;
      options.split_on_gap = std::chrono::milliseconds(config.recording.split_on_gap_ms);
      auto& recorder = assembly.Add<dsf::record::SegmentRecorder>(options, std::move(events));
      by_source[static_cast<std::size_t>(slot.source)].push_back(&recorder);
    }

    dsf::ds::MetaTap::Options options;
    options.element = example_app::kMetaTap;
    options.width = config.mux.width;
    options.height = config.mux.height;
    options.analytics = config.analytics.enabled;
    options.image = config.redis.image;  // 默认对有目标的帧编码
    options.gpu_id = config.mux.gpu_id;
    auto& tap = assembly.Add<dsf::ds::MetaTap>(options, sources);
    if (!config.inference.empty()) {
      const bool only_on_detection = config.redis.only_on_detection;
      tap.OnBatch([detections, by_source, only_on_detection](const dsf::ds::BatchRecord& batch) {
        for (const auto& frame : batch.frames) {
          if (!frame.objects.empty() && frame.pad_index < by_source.size()) {
            for (auto* recorder : by_source[frame.pad_index]) recorder->MarkDetected();
          }
          if (detections != nullptr && (!only_on_detection || !frame.objects.empty())) {
            detections->TryPublish(dsf::ds::DetectionEntry(frame));
          }
        }
      });
    }

    // 就绪 = 真实的帧到达了元数据出口（按批次元数据中的帧计数），而不只是进入 PLAYING。
    assembly.ReadyWhen([&tap] { return tap.frames(); });
  }
};

}  // namespace

int main(int argc, char** argv) { return dsf::app::Main<ExampleApp>(argc, argv); }
