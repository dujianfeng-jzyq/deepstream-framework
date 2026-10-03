// 元数据采集（需要 DeepStream SDK）：把 NvDsBatchMeta 转换为 dsf::ds::BatchRecord。
//
// 探针挂在 AddMetaTap 创建的 identity 的 src pad 上。转换在媒体线程完成，
// 之后的处理有两种：
//   OnBatch        同步，在媒体线程执行，必须快（打录像标记、入队消息）
//   OnBatchAsync   框架复制记录并投递到工作线程，队列满时丢弃并计数
// 提取器把框架不认识的元数据（自定义 user meta 等）转换为 FrameRecord/ObjectRecord 的 extra。

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
#include <vector>

#include <gst/gst.h>
#include <nvdsmeta.h>

#include "dsf/ds/meta.h"
#include "dsf/gst/component.h"

namespace dsf::ds {

class MetaTap : public gst::Component {
 public:
  using Handler = std::function<void(const BatchRecord&)>;
  using FrameExtractor = std::function<void(const NvDsFrameMeta&, FrameRecord&)>;
  using ObjectExtractor = std::function<void(const NvDsObjectMeta&, ObjectRecord&)>;

  struct Options {
    std::string element;  // AddMetaTap 的元件名
    int width = 0;        // mux 输出尺寸，框坐标所在的空间
    int height = 0;
    bool classifiers = true;  // 采集二级分类结果
    bool analytics = false;   // 采集 nvdsanalytics 结果（帧级计数与目标状态）
  };

  MetaTap(Options options, SourceRegistry sources);
  ~MetaTap() override;

  void OnBatch(Handler handler);
  void OnBatchAsync(Handler handler, std::size_t capacity = 64);
  void AddFrameExtractor(FrameExtractor extractor);
  void AddObjectExtractor(ObjectExtractor extractor);

  std::string_view Name() const override { return name_; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  void Report(gst::Stats& stats) const override;

  std::uint64_t frames() const { return frames_.load(); }

 private:
  struct Async;
  static GstPadProbeReturn OnBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  void Handle(GstBuffer* buffer);
  BatchRecord Convert(NvDsBatchMeta* batch) const;

  Options options_;
  SourceRegistry sources_;
  std::string name_;
  std::vector<Handler> handlers_;
  std::vector<std::unique_ptr<Async>> async_;
  std::vector<FrameExtractor> frame_extractors_;
  std::vector<ObjectExtractor> object_extractors_;
  GstPad* pad_ = nullptr;
  gulong probe_ = 0;
  std::atomic<std::uint64_t> frames_{0};
  std::atomic<bool> warned_missing_{false};
};

// 运行环境中的 DeepStream 版本（读取 SDK 的 version 文件），读不到返回 "unknown"。
std::string DeepStreamVersion();

}  // namespace dsf::ds

#endif  // DSF_DS_META_TAP_H
