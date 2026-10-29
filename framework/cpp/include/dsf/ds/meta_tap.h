// 元数据采集（需要 DeepStream SDK）：把 NvDsBatchMeta 转换为 dsf::ds::BatchRecord。
//
// 探针挂在 AddMetaTap 创建的 identity 的 src pad 上。转换在媒体线程完成，
// 之后的处理有两种：
//   OnBatch        同步，在媒体线程执行，必须快（打录像标记、入队消息）
//   OnBatchAsync   框架复制记录并投递到工作线程，队列满时丢弃并计数
// 提取器把框架不认识的元数据（自定义 user meta 等）转换为 FrameRecord/ObjectRecord 的 extra。
//
// image.enabled 时，在转换之后、处理函数之前，用 DeepStream 的批量编码库（nvds_obj_enc：Jetson 上是
// NVJPG 硬件，dGPU 上是 nvJPEG/CUDA）把选中的帧整帧编码为 JPEG，直接读取 GPU 上的批次 surface，
// 数据复制进 FrameRecord::image。编码在采集点的流线程同步完成（每批等待一次 GPU），
// AddMetaTap 用同一个 ImageSpec 在采集点前插入 queue，使编码与上游推理并行。
// 编码失败只计数，该帧照常交给处理函数，image 为空。

#ifndef DSF_DS_META_TAP_H
#define DSF_DS_META_TAP_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gst/gst.h>
#include <nvdsmeta.h>

#include "dsf/ds/blocks.h"
#include "dsf/ds/meta.h"
#include "dsf/gst/component.h"
#include "dsf/gst/probe.h"

namespace dsf::ds {

class MetaTap : public gst::Component {
 public:
  using Handler = std::function<void(const BatchRecord&)>;
  using FrameExtractor = std::function<void(const NvDsFrameMeta&, FrameRecord&)>;
  using ObjectExtractor = std::function<void(const NvDsObjectMeta&, ObjectRecord&)>;
  using ImageFilter = std::function<bool(const FrameRecord&)>;

  struct Options {
    std::string element;  // AddMetaTap 的元件名
    int width = 0;        // mux 输出尺寸，框坐标所在的空间
    int height = 0;
    bool classifiers = true;  // 采集二级分类结果
    bool analytics = false;   // 采集 nvdsanalytics 结果（帧级计数与目标状态）
    ImageSpec image;          // 整帧图片编码；默认关闭
    int gpu_id = 0;           // 编码使用的 GPU（与 mux 的 gpu_id 一致）

    // 采集点接在 mux 之后时的选项：框坐标空间与编码 GPU 取自 mux，不必逐项抄写。
    static Options ForMux(std::string element, const MuxSpec& mux) {
      Options options;
      options.element = std::move(element);
      options.width = mux.width;
      options.height = mux.height;
      options.gpu_id = mux.gpu_id;
      return options;
    }
  };

  MetaTap(Options options, SourceRegistry sources);
  ~MetaTap() override;

  void OnBatch(Handler handler);
  void OnBatchAsync(Handler handler, std::size_t capacity = 64);
  void AddFrameExtractor(FrameExtractor extractor);
  void AddObjectExtractor(ObjectExtractor extractor);
  // 选择要编码图片的帧（媒体线程，必须快）；默认选择有目标的帧。只在 image.enabled 时生效。
  void SetImageFilter(ImageFilter filter);

  std::string_view Name() const override { return name_; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  void Report(gst::Stats& stats) const override;

  std::uint64_t frames() const { return frames_.load(); }

 private:
  struct Async;
  struct Encoder;
  static GstPadProbeReturn OnBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  void Handle(GstBuffer* buffer);
  BatchRecord Convert(NvDsBatchMeta* batch, std::vector<NvDsFrameMeta*>& metas) const;
  void Encode(GstBuffer* buffer, BatchRecord& record, const std::vector<NvDsFrameMeta*>& metas);

  Options options_;
  SourceRegistry sources_;
  std::string name_;
  std::vector<Handler> handlers_;
  std::vector<std::unique_ptr<Async>> async_;
  std::vector<FrameExtractor> frame_extractors_;
  std::vector<ObjectExtractor> object_extractors_;
  ImageFilter image_filter_;
  std::unique_ptr<Encoder> encoder_;  // OnCreated 加载库并创建上下文，OnReleased 释放
  std::atomic<std::uint64_t> frames_{0};
  std::atomic<bool> warned_missing_{false};
  std::atomic<std::uint64_t> images_{0};
  std::atomic<std::uint64_t> image_failures_{0};
  std::atomic<std::uint64_t> encode_batches_{0};
  std::atomic<std::uint64_t> encode_us_{0};
  gst::PadProbe probe_;
};

// 运行环境中的 DeepStream 版本（读取 SDK 的 version 文件），读不到返回 "unknown"。
std::string DeepStreamVersion();

}  // namespace dsf::ds

#endif  // DSF_DS_META_TAP_H
