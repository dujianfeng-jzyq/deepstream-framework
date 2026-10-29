// pad 探针句柄的实现。

#include "dsf/gst/probe.h"

#include <stdexcept>

namespace dsf::gst {

void PadProbe::Attach(GstBin* bin, const std::string& element, const std::string& pad, GstPadProbeType mask,
                      GstPadProbeCallback callback, gpointer data) {
  GstElement* found = gst_bin_get_by_name(bin, element.c_str());
  if (found == nullptr) throw std::runtime_error("element '" + element + "' is not in the pipeline");
  GstPad* static_pad = gst_element_get_static_pad(found, pad.c_str());
  gst_object_unref(found);
  if (static_pad == nullptr) throw std::runtime_error("element '" + element + "' has no " + pad + " pad");
  Attach(static_pad, mask, callback, data);
  gst_object_unref(static_pad);
}

void PadProbe::Attach(GstPad* pad, GstPadProbeType mask, GstPadProbeCallback callback, gpointer data) {
  Remove();
  pad_ = GST_PAD(gst_object_ref(pad));
  id_ = gst_pad_add_probe(pad_, mask, callback, data, nullptr);
}

void PadProbe::Remove() noexcept {
  if (pad_ == nullptr) return;
  if (id_ != 0) gst_pad_remove_probe(pad_, id_);
  id_ = 0;
  gst_object_unref(pad_);
  pad_ = nullptr;
}

}  // namespace dsf::gst
