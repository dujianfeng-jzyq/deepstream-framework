// pad 探针句柄：持有 pad 的引用，Remove() 或析构时拆除探针并释放 pad。
//
// 组件在 OnCreated 中 Attach，在 OnReleased 中 Remove：那时管线已置 NULL、流线程已停止。
// gst_pad_remove_probe 不等待正在执行的回调，流线程仍在运行时拆除，回调可能访问已释放的数据。
// 句柄应声明在回调数据之后，使析构时先拆探针、后释放数据。

#ifndef DSF_GST_PROBE_H
#define DSF_GST_PROBE_H

#include <string>

#include <gst/gst.h>

namespace dsf::gst {

class PadProbe {
 public:
  PadProbe() = default;
  ~PadProbe() { Remove(); }
  PadProbe(const PadProbe&) = delete;
  PadProbe& operator=(const PadProbe&) = delete;

  // 挂在 bin 中名为 element 的元件的静态 pad 上；元件或 pad 不存在时抛 std::runtime_error。
  // 已挂载时先拆除原探针。
  void Attach(GstBin* bin, const std::string& element, const std::string& pad, GstPadProbeType mask,
              GstPadProbeCallback callback, gpointer data);
  // 挂在给定 pad 上；另取引用，不接管调用方的引用。
  void Attach(GstPad* pad, GstPadProbeType mask, GstPadProbeCallback callback, gpointer data);
  void Remove() noexcept;

  // 借用指针，未挂载时为空。
  GstPad* pad() const { return pad_; }

 private:
  GstPad* pad_ = nullptr;
  gulong id_ = 0;
};

}  // namespace dsf::gst

#endif  // DSF_GST_PROBE_H
