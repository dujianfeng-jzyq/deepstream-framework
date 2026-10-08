// 元数据采集的实现。

#include "dsf/ds/meta_tap.h"

// DeepStream 7.1 的 nvds_analytics_meta.h 使用 std::string/std::unordered_map 却没有包含对应头文件。
#include <string>
#include <unordered_map>

#include <dlfcn.h>
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include <nvds_analytics_meta.h>
#include <nvds_obj_encode.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>

#include "dsf/gst/pipeline.h"

namespace dsf::ds {

// Jetson 的 surface/transform 库由 NVIDIA runtime 提供，无 GPU 的构建和自检不应加载它们。
// 按 soname 加载：由可执行文件的 RUNPATH（dsf::deepstream 写入的 SDK lib 目录）解析，
// 不把构建机解析出的版本目录写进载荷。x86 镜像的 ld.so.conf 不含 SDK 目录，RUNPATH 不可省。
// 库必须存活到编码上下文销毁之后；构造失败时也由 RAII 释放已打开的句柄。
struct MetaTap::Encoder {
  explicit Encoder(int gpu_id) : library_(Open()) {
    auto create = Symbol<decltype(&nvds_obj_enc_create_context)>("nvds_obj_enc_create_context");
    destroy_ = Symbol<decltype(&nvds_obj_enc_destroy_context)>("nvds_obj_enc_destroy_context");
    process_ = Symbol<decltype(&nvds_obj_enc_process)>("nvds_obj_enc_process");
    finish_ = Symbol<decltype(&nvds_obj_enc_finish)>("nvds_obj_enc_finish");
    context_ = create(gpu_id);
    if (context_ == nullptr) {
      throw std::runtime_error("cannot create image encoder on GPU " + std::to_string(gpu_id));
    }
  }

  ~Encoder() { destroy_(context_); }

  bool Process(NvDsObjEncUsrArgs* args, NvBufSurface* surface, NvDsFrameMeta* frame) {
    return process_(context_, args, surface, nullptr, frame);
  }

  void Finish() { finish_(context_); }

 private:
  static void* Open() {
    void* library = dlopen("libnvds_batch_jpegenc.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      throw std::runtime_error(std::string("cannot load image encoder: ") + dlerror() +
                               "; image encoding requires NVIDIA runtime libraries");
    }
    return library;
  }

  template <typename Function>
  Function Symbol(const char* name) {
    dlerror();
    void* symbol = dlsym(library_.get(), name);
    if (const char* error = dlerror()) {
      throw std::runtime_error(std::string("cannot resolve image encoder symbol ") + name + ": " + error);
    }
    return reinterpret_cast<Function>(symbol);
  }

  struct LibraryCloser {
    void operator()(void* library) const { dlclose(library); }
  };

  std::unique_ptr<void, LibraryCloser> library_;
  NvDsObjEncCtxHandle context_ = nullptr;
  decltype(&nvds_obj_enc_destroy_context) destroy_ = nullptr;
  decltype(&nvds_obj_enc_process) process_ = nullptr;
  decltype(&nvds_obj_enc_finish) finish_ = nullptr;
};

// 有界队列 + 单工作线程；记录按值入队，媒体线程不等待处理。
struct MetaTap::Async {
  Handler handler;
  std::size_t capacity;
  std::mutex mutex;
  std::condition_variable wakeup;
  std::deque<BatchRecord> queue;
  bool stopping = false;
  std::thread worker;
  std::atomic<std::uint64_t> dropped{0};

  void Start() {
    worker = std::thread([this] {
      for (;;) {
        BatchRecord batch;
        {
          std::unique_lock<std::mutex> lock(mutex);
          wakeup.wait(lock, [this] { return stopping || !queue.empty(); });
          if (queue.empty()) return;  // 停止时先处理完已入队的记录
          batch = std::move(queue.front());
          queue.pop_front();
        }
        try {
          handler(batch);
        } catch (const std::exception& error) {
          spdlog::error("metadata handler failed: {}", error.what());
        }
      }
    });
  }
  void Push(const BatchRecord& batch) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (queue.size() >= capacity) {
        ++dropped;
        return;
      }
      queue.push_back(batch);
    }
    wakeup.notify_one();
  }
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    wakeup.notify_one();
    if (worker.joinable()) worker.join();
  }
};

MetaTap::MetaTap(Options options, SourceRegistry sources)
    : options_(std::move(options)), sources_(std::move(sources)), name_("meta-tap:" + options_.element) {}

MetaTap::~MetaTap() { OnReleased(); }

void MetaTap::OnBatch(Handler handler) { handlers_.push_back(std::move(handler)); }

void MetaTap::OnBatchAsync(Handler handler, std::size_t capacity) {
  auto async = std::make_unique<Async>();
  async->handler = std::move(handler);
  async->capacity = capacity == 0 ? 1 : capacity;
  async_.push_back(std::move(async));
}

void MetaTap::AddFrameExtractor(FrameExtractor extractor) {
  frame_extractors_.push_back(std::move(extractor));
}

void MetaTap::AddObjectExtractor(ObjectExtractor extractor) {
  object_extractors_.push_back(std::move(extractor));
}

void MetaTap::SetImageFilter(ImageFilter filter) { image_filter_ = std::move(filter); }

void MetaTap::OnCreated(gst::Pipeline& /*pipeline*/, GstElement* bin) {
  GstElement* element = gst_bin_get_by_name(GST_BIN(bin), options_.element.c_str());
  if (element == nullptr) {
    throw std::runtime_error("metadata tap element '" + options_.element + "' is not in the pipeline");
  }
  pad_ = gst_element_get_static_pad(element, "src");
  gst_object_unref(element);
  if (pad_ == nullptr) throw std::runtime_error("element '" + options_.element + "' has no src pad");
  if (options_.image.enabled) {
    encoder_ = std::make_unique<Encoder>(options_.gpu_id);
  }
  for (auto& async : async_) async->Start();
  probe_ = gst_pad_add_probe(pad_, GST_PAD_PROBE_TYPE_BUFFER, &MetaTap::OnBuffer, this, nullptr);
}

void MetaTap::OnReleased() {
  if (pad_ != nullptr) {
    if (probe_ != 0) gst_pad_remove_probe(pad_, probe_);
    probe_ = 0;
    gst_object_unref(pad_);
    pad_ = nullptr;
  }
  // 管线已置 NULL、流线程已停止，不会再有编码在进行。
  encoder_.reset();
  for (auto& async : async_) async->Stop();
}

void MetaTap::Report(gst::Stats& stats) const {
  stats.Add(options_.element + "_frames", frames_.load());
  std::uint64_t dropped = 0;
  for (const auto& async : async_) dropped += async->dropped.load();
  if (!async_.empty()) stats.Add(options_.element + "_async_dropped", dropped);
  if (options_.image.enabled) {
    stats.Add(options_.element + "_images", images_.load());
    stats.Add(options_.element + "_image_failures", image_failures_.load());
    const std::uint64_t batches = encode_batches_.load();
    stats.Add(options_.element + "_encode_us_avg", batches == 0 ? 0 : encode_us_.load() / batches);
  }
}

GstPadProbeReturn MetaTap::OnBuffer(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer self) {
  auto* tap = static_cast<MetaTap*>(self);
  try {
    if (GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info)) tap->Handle(buffer);
  } catch (const std::exception& error) {
    // 异常不能穿过 C 回调；记录后继续处理视频。
    spdlog::error("metadata tap {} failed: {}", tap->options_.element, error.what());
  }
  return GST_PAD_PROBE_OK;
}

void MetaTap::Handle(GstBuffer* buffer) {
  NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buffer);
  if (batch == nullptr) {
    if (!warned_missing_.exchange(true)) {
      spdlog::warn("no DeepStream batch metadata at '{}'; it must be downstream of nvstreammux",
                   options_.element);
    }
    return;
  }
  std::vector<NvDsFrameMeta*> metas;
  BatchRecord record = Convert(batch, metas);
  frames_ += record.frames.size();
  if (encoder_ != nullptr) Encode(buffer, record, metas);
  for (const Handler& handler : handlers_) handler(record);
  for (auto& async : async_) async->Push(record);
}

namespace {

void ReadAnalyticsFrame(const NvDsFrameMeta& frame, FrameRecord& record) {
  for (NvDsMetaList* item = frame.frame_user_meta_list; item != nullptr; item = item->next) {
    auto* user = static_cast<NvDsUserMeta*>(item->data);
    if (user == nullptr || user->base_meta.meta_type != NVDS_USER_FRAME_META_NVDSANALYTICS) continue;
    const auto* meta = static_cast<NvDsAnalyticsFrameMeta*>(user->user_meta_data);
    if (meta == nullptr) continue;
    nlohmann::json analytics;
    analytics["roi_counts"] = meta->objInROIcnt;
    analytics["line_crossing_current"] = meta->objLCCurrCnt;
    analytics["line_crossing_total"] = meta->objLCCumCnt;
    analytics["overcrowding"] = meta->ocStatus;
    record.extra["analytics"] = std::move(analytics);
  }
}

void ReadAnalyticsObject(const NvDsObjectMeta& object, ObjectRecord& record) {
  for (NvDsMetaList* item = object.obj_user_meta_list; item != nullptr; item = item->next) {
    auto* user = static_cast<NvDsUserMeta*>(item->data);
    if (user == nullptr || user->base_meta.meta_type != NVDS_USER_OBJ_META_NVDSANALYTICS) continue;
    const auto* info = static_cast<NvDsAnalyticsObjInfo*>(user->user_meta_data);
    if (info == nullptr) continue;
    record.extra["analytics"] = {{"roi", info->roiStatus},
                                 {"line_crossing", info->lcStatus},
                                 {"overcrowding", info->ocStatus},
                                 {"direction", info->dirStatus}};
  }
}

void ReadClassifiers(const NvDsObjectMeta& object, ObjectRecord& record) {
  for (NvDsMetaList* item = object.classifier_meta_list; item != nullptr; item = item->next) {
    const auto* classifier = static_cast<NvDsClassifierMeta*>(item->data);
    if (classifier == nullptr) continue;
    for (NvDsMetaList* label = classifier->label_info_list; label != nullptr; label = label->next) {
      const auto* info = static_cast<NvDsLabelInfo*>(label->data);
      if (info == nullptr) continue;
      Classification result;
      result.component_id = classifier->unique_component_id;
      result.class_id = static_cast<int>(info->result_class_id);
      result.label = info->pResult_label != nullptr ? info->pResult_label : info->result_label;
      result.confidence = info->result_prob;
      record.classes.push_back(std::move(result));
    }
  }
}

std::vector<NvDsUserMeta*> ImageMetas(const NvDsFrameMeta& frame) {
  std::vector<NvDsUserMeta*> metas;
  for (NvDsMetaList* item = frame.frame_user_meta_list; item != nullptr; item = item->next) {
    auto* user = static_cast<NvDsUserMeta*>(item->data);
    if (user != nullptr && user->base_meta.meta_type == NVDS_CROP_IMAGE_META) metas.push_back(user);
  }
  return metas;
}

}  // namespace

// 编码结果由库以 NVDS_CROP_IMAGE_META 挂到帧上。上游的其他采集点也可能挂过同类元数据，
// 因此按编码前后的差异找到本次的结果；复制数据后立即移除，编码缓冲随之归还，不随批次流向下游。
void MetaTap::Encode(GstBuffer* buffer, BatchRecord& record, const std::vector<NvDsFrameMeta*>& metas) {
  struct Job {
    std::size_t index;
    std::vector<NvDsUserMeta*> before;
    int width;
    int height;
  };
  std::vector<Job> jobs;
  for (std::size_t i = 0; i < record.frames.size(); ++i) {
    const FrameRecord& frame = record.frames[i];
    if (image_filter_ ? image_filter_(frame) : !frame.objects.empty()) jobs.push_back({i, {}, 0, 0});
  }
  if (jobs.empty()) return;

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    image_failures_ += jobs.size();
    spdlog::warn("cannot map batch buffer at '{}' for image encoding", options_.element);
    return;
  }
  auto* surface = reinterpret_cast<NvBufSurface*>(map.data);
  const ImageSpec& spec = options_.image;
  const auto begin = std::chrono::steady_clock::now();
  std::vector<Job> submitted;
  for (Job& job : jobs) {
    NvDsFrameMeta* meta = metas[job.index];
    if (meta->batch_id >= surface->numFilled) {
      ++image_failures_;
      continue;
    }
    const NvBufSurfaceParams& params = surface->surfaceList[meta->batch_id];
    job.width = spec.width > 0 ? spec.width : static_cast<int>(params.width);
    job.height = spec.height > 0 ? spec.height : static_cast<int>(params.height);
    job.before = ImageMetas(*meta);
    NvDsObjEncUsrArgs args{};
    args.isFrame = true;
    args.attachUsrMeta = true;
    args.saveImg = false;
    args.quality = spec.quality;
    if (spec.width > 0 && spec.height > 0) {
      args.scaleImg = true;
      args.scaledWidth = spec.width;
      args.scaledHeight = spec.height;
    }
    if (encoder_->Process(&args, surface, meta)) {
      submitted.push_back(std::move(job));
    } else {
      ++image_failures_;
    }
  }
  // 等待本批全部编码完成；之后才能读取结果。
  encoder_->Finish();
  gst_buffer_unmap(buffer, &map);
  const auto elapsed = std::chrono::steady_clock::now() - begin;
  encode_us_ += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
  ++encode_batches_;

  for (const Job& job : submitted) {
    NvDsFrameMeta* meta = metas[job.index];
    NvDsUserMeta* produced = nullptr;
    for (NvDsUserMeta* user : ImageMetas(*meta)) {
      if (std::find(job.before.begin(), job.before.end(), user) == job.before.end()) produced = user;
    }
    const auto* out = produced != nullptr ? static_cast<NvDsObjEncOutParams*>(produced->user_meta_data) : nullptr;
    if (out == nullptr || out->outBuffer == nullptr || out->outLen == 0) {
      ++image_failures_;
    } else {
      Image image;
      image.format = "jpeg";
      image.width = job.width;
      image.height = job.height;
      image.bytes.assign(reinterpret_cast<const char*>(out->outBuffer), out->outLen);
      record.frames[job.index].image = std::move(image);
      ++images_;
    }
    if (produced != nullptr) nvds_remove_user_meta_from_frame(meta, produced);
  }
}

BatchRecord MetaTap::Convert(NvDsBatchMeta* batch, std::vector<NvDsFrameMeta*>& metas) const {
  BatchRecord record;
  const std::uint64_t now = WallMilliseconds();
  for (NvDsMetaList* f = batch->frame_meta_list; f != nullptr; f = f->next) {
    auto* frame = static_cast<NvDsFrameMeta*>(f->data);
    if (frame == nullptr) continue;
    FrameRecord out;
    out.pad_index = frame->pad_index;
    out.source_id = sources_.IdOf(frame->pad_index);
    out.frame_num = static_cast<std::uint64_t>(frame->frame_num);
    out.pts_ns = frame->buf_pts;
    out.ntp_ms = frame->ntp_timestamp / 1000000;
    out.wall_ms = now;
    out.width = options_.width;
    out.height = options_.height;
    for (NvDsMetaList* o = frame->obj_meta_list; o != nullptr; o = o->next) {
      const auto* object = static_cast<NvDsObjectMeta*>(o->data);
      if (object == nullptr) continue;
      ObjectRecord item;
      item.class_id = object->class_id;
      item.label = object->obj_label;
      item.confidence = object->confidence;
      item.component_id = object->unique_component_id;
      item.box = {object->rect_params.left, object->rect_params.top, object->rect_params.width,
                  object->rect_params.height};
      if (object->object_id != UNTRACKED_OBJECT_ID) item.track_id = object->object_id;
      if (options_.classifiers) ReadClassifiers(*object, item);
      if (options_.analytics) ReadAnalyticsObject(*object, item);
      for (const auto& extractor : object_extractors_) extractor(*object, item);
      out.objects.push_back(std::move(item));
    }
    if (options_.analytics) ReadAnalyticsFrame(*frame, out);
    for (const auto& extractor : frame_extractors_) extractor(*frame, out);
    record.frames.push_back(std::move(out));
    metas.push_back(frame);
  }
  return record;
}

std::string DeepStreamVersion() {
  std::ifstream stream(DSF_DEEPSTREAM_VERSION_FILE);
  std::string line;
  std::getline(stream, line);
  // 文件内容形如 "Version: 8.0"。
  const std::size_t colon = line.find(':');
  if (colon != std::string::npos) line = line.substr(colon + 1);
  const std::size_t begin = line.find_first_not_of(" \t\r");
  const std::size_t end = line.find_last_not_of(" \t\r");
  return begin == std::string::npos ? "unknown" : line.substr(begin, end - begin + 1);
}

}  // namespace dsf::ds
