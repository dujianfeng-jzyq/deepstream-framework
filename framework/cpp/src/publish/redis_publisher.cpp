// Redis Stream 发布器：一个连接、一个工作线程、多个通道。
// 每个通道有独立队列，检测消息的洪峰不会挤掉录像事件；入队即唤醒发送，积压时每轮以 pipelining 合并发送。
// 每轮同时受条数与字节上限约束：带图片的大消息一轮发得太多会让 Redis 单线程长时间卡住。

#include <hiredis/hiredis.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "dsf/publish/publisher.h"

namespace dsf::publish {
namespace {

using Clock = std::chrono::steady_clock;

timeval ToTimeval(int milliseconds) {
  timeval value{};
  value.tv_sec = milliseconds / 1000;
  value.tv_usec = (milliseconds % 1000) * 1000;
  return value;
}

std::size_t SizeOf(const Fields& fields) {
  std::size_t size = 0;
  for (const auto& [name, value] : fields) size += name.size() + value.size();
  return size;
}

class RedisPublisher;

class RedisChannel : public Channel {
 public:
  RedisChannel(RedisPublisher& owner, ChannelSpec spec) : owner_(owner), spec_(std::move(spec)) {}
  using Channel::TryPublish;
  bool TryPublish(Fields fields) override;
  ChannelStats Stats() const override;
  const std::string& target() const override { return spec_.target; }

 private:
  friend class RedisPublisher;
  RedisPublisher& owner_;
  ChannelSpec spec_;
  std::deque<Fields> queue_;  // 由 owner_.mutex_ 保护
  std::size_t queued_bytes_ = 0;
  std::atomic<std::uint64_t> published_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<std::uint64_t> failed_{0};
};

class RedisPublisher : public Publisher {
 public:
  explicit RedisPublisher(RedisOptions options) : options_(std::move(options)) {
    if (options_.batch_size <= 0 || options_.batch_bytes <= 0) {
      throw std::invalid_argument("batch_size and batch_bytes must be positive");
    }
  }
  ~RedisPublisher() override { Stop(); }

  Channel& Open(const ChannelSpec& spec) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::kIdle) throw std::logic_error("channels must be opened before Start");
    if (spec.target.empty()) throw std::invalid_argument("channel target must not be empty");
    if (spec.capacity <= 0 || spec.max_length <= 0) {
      throw std::invalid_argument("channel capacity and max_length must be positive");
    }
    channels_.push_back(std::make_unique<RedisChannel>(*this, spec));
    return *channels_.back();
  }

  void Start() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::kIdle) return;  // 一次性：停止后不能再次启动
    state_ = State::kRunning;
    worker_ = std::thread([this] { Work(); });
  }

  void Stop() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ != State::kRunning) return;  // 未启动、正在或已经停止
      state_ = State::kDraining;
      deadline_ = Clock::now() + std::chrono::milliseconds(options_.drain_timeout_ms);
    }
    wakeup_.notify_all();
    worker_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = State::kStopped;
  }

  bool connected() const override { return connected_.load(); }

  std::vector<const Channel*> Channels() const override {
    std::vector<const Channel*> channels;
    for (const auto& channel : channels_) channels.push_back(channel.get());
    return channels;
  }

  std::string Describe() const override {
    std::string text = "redis " + options_.host + ":" + std::to_string(options_.port) + " streams=";
    for (std::size_t i = 0; i < channels_.size(); ++i) {
      text += (i == 0 ? "" : ",") + channels_[i]->target();
    }
    return text;
  }

  bool Enqueue(RedisChannel& channel, Fields fields) {
    const std::size_t size = SizeOf(fields);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const std::size_t max_bytes = channel.spec_.max_bytes;
      // 字节上限只约束积压：空队列总能放下一条，单条超过上限的消息不会永远发不出去。
      const bool fits = max_bytes == 0 || channel.queue_.empty() || channel.queued_bytes_ + size <= max_bytes;
      if (state_ == State::kRunning && !fields.empty() && fits &&
          channel.queue_.size() < static_cast<std::size_t>(channel.spec_.capacity)) {
        channel.queue_.push_back(std::move(fields));
        channel.queued_bytes_ += size;
        wakeup_.notify_one();
        return true;
      }
    }
    ++channel.dropped_;
    Warn("queue full or publisher not running; dropping a message for " + channel.target());
    return false;
  }

  std::uint64_t Queued(const RedisChannel& channel) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return channel.queue_.size();
  }

 private:
  using Batch = std::vector<std::pair<RedisChannel*, Fields>>;

  bool HasWork() const {
    for (const auto& channel : channels_) {
      if (!channel->queue_.empty()) return true;
    }
    return false;
  }

  // 轮流从各通道取消息，避免某个通道独占一轮。超过字节上限时停止，但每轮至少取一条。
  Batch Take() {
    Batch batch;
    std::size_t bytes = 0;
    const auto full = [&] {
      return batch.size() >= static_cast<std::size_t>(options_.batch_size) ||
             bytes >= static_cast<std::size_t>(options_.batch_bytes);
    };
    bool progressed = true;
    while (progressed && !full()) {
      progressed = false;
      for (const auto& channel : channels_) {
        if (channel->queue_.empty() || full()) continue;
        const std::size_t size = SizeOf(channel->queue_.front());
        if (!batch.empty() && bytes + size > static_cast<std::size_t>(options_.batch_bytes)) continue;
        batch.emplace_back(channel.get(), std::move(channel->queue_.front()));
        channel->queue_.pop_front();
        channel->queued_bytes_ -= size;
        bytes += size;
        progressed = true;
      }
    }
    return batch;
  }

  void Work() {
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wakeup_.wait(lock, [this] { return state_ == State::kDraining || HasWork(); });
        if (state_ == State::kDraining && (!HasWork() || Clock::now() >= deadline_)) break;
      }
      if (!Connect()) {
        // 断线时不取出消息：队列继续缓冲，满了才在入队处丢弃。
        std::unique_lock<std::mutex> lock(mutex_);
        const auto retry = Clock::now() + std::chrono::seconds(1);
        const bool draining = state_ == State::kDraining;
        wakeup_.wait_until(lock, draining ? std::min(retry, deadline_) : retry,
                           [this] { return state_ == State::kDraining && Clock::now() >= deadline_; });
        continue;
      }
      Batch batch;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        batch = Take();
      }
      Send(batch);
    }
    // 排空期限已到：剩余消息计为丢弃，让发出与丢失的计数能够对账。
    std::uint64_t remaining = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& channel : channels_) {
        channel->dropped_ += channel->queue_.size();
        remaining += channel->queue_.size();
        channel->queue_.clear();
        channel->queued_bytes_ = 0;
      }
    }
    if (remaining > 0) spdlog::warn("publisher stopped with {} unsent message(s)", remaining);
    Disconnect();
  }

  bool Connect() {
    if (context_ != nullptr) return true;
    context_ = redisConnectWithTimeout(options_.host.c_str(), options_.port,
                                       ToTimeval(options_.connect_timeout_ms));
    if (context_ == nullptr || context_->err != 0) {
      Warn("cannot connect to redis " + options_.host + ":" + std::to_string(options_.port) + ": " +
           (context_ != nullptr ? context_->errstr : "allocation failed"));
      Disconnect();
      return false;
    }
    redisSetTimeout(context_, ToTimeval(options_.command_timeout_ms));
    if (!options_.password.empty()) {
      auto* reply = static_cast<redisReply*>(redisCommand(context_, "AUTH %s", options_.password.c_str()));
      const bool ok = reply != nullptr && reply->type != REDIS_REPLY_ERROR;
      if (reply != nullptr) freeReplyObject(reply);
      if (!ok) {
        Warn("redis AUTH failed for " + options_.host);
        Disconnect();
        return false;
      }
    }
    connected_ = true;
    spdlog::info("publisher connected: {}", Describe());
    return true;
  }

  void Disconnect() {
    if (context_ != nullptr) redisFree(context_);
    context_ = nullptr;
    connected_ = false;
  }

  void Send(Batch& batch) {
    std::vector<const char*> argv;
    std::vector<std::size_t> lengths;
    for (auto& [channel, fields] : batch) {
      // 用 argv 形式按长度传参：字段值可以是任意二进制（例如 JPEG），hiredis 不做转义与复制之外的处理。
      const std::string max_length = std::to_string(channel->spec_.max_length);
      argv = {"XADD", channel->target().c_str(), "MAXLEN", "~", max_length.c_str(), "*"};
      lengths = {4, channel->target().size(), 6, 1, max_length.size(), 1};
      for (const auto& [name, value] : fields) {
        argv.push_back(name.data());
        lengths.push_back(name.size());
        argv.push_back(value.data());
        lengths.push_back(value.size());
      }
      redisAppendCommandArgv(context_, static_cast<int>(argv.size()), argv.data(), lengths.data());
    }
    for (std::size_t i = 0; i < batch.size(); ++i) {
      RedisChannel& channel = *batch[i].first;
      void* raw = nullptr;
      if (redisGetReply(context_, &raw) != REDIS_OK) {
        // 连接中断：本条及之后未确认的消息都计为失败，下一轮重新连接。
        const std::size_t lost = batch.size() - i;
        for (std::size_t j = i; j < batch.size(); ++j) ++batch[j].first->failed_;
        Warn("redis connection lost; " + std::to_string(lost) + " message(s) not confirmed");
        Disconnect();
        return;
      }
      auto* reply = static_cast<redisReply*>(raw);
      if (reply->type == REDIS_REPLY_ERROR) {
        ++channel.failed_;
        Warn("redis rejected XADD to " + channel.target() + ": " + std::string(reply->str, reply->len));
      } else {
        ++channel.published_;
      }
      freeReplyObject(reply);
    }
  }

  // 断线时每条消息都会失败，限频避免日志被刷满。
  void Warn(const std::string& message) {
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         Clock::now().time_since_epoch()).count();
    std::int64_t last = last_warning_ms_.load();
    if (now - last < 5000 || !last_warning_ms_.compare_exchange_strong(last, now)) return;
    spdlog::warn("{}", message);
  }

  // 只进不退：kIdle 只能打开通道，kRunning 接收消息，kDraining 在期限内发完剩余消息，kStopped 结束。
  enum class State { kIdle, kRunning, kDraining, kStopped };

  RedisOptions options_;
  mutable std::mutex mutex_;
  std::condition_variable wakeup_;
  std::vector<std::unique_ptr<RedisChannel>> channels_;
  State state_ = State::kIdle;  // 由 mutex_ 保护
  Clock::time_point deadline_;
  std::thread worker_;
  redisContext* context_ = nullptr;  // 只在工作线程使用
  std::atomic<bool> connected_{false};
  std::atomic<std::int64_t> last_warning_ms_{-1000000};
};

bool RedisChannel::TryPublish(Fields fields) { return owner_.Enqueue(*this, std::move(fields)); }

ChannelStats RedisChannel::Stats() const {
  return {published_.load(), dropped_.load(), failed_.load(), owner_.Queued(*this)};
}

}  // namespace

RedisOptions RedisOptions::Read(config::Reader reader) {
  RedisOptions options;
  options.host = reader.Get<std::string>("host", "127.0.0.1", config::NonEmpty());
  options.port = reader.Get("port", 6379, config::Range(1, 65535));
  options.password = reader.Secret("password");
  options.connect_timeout_ms = reader.Get("connect_timeout_ms", 1000, config::AtLeast(1));
  options.command_timeout_ms = reader.Get("command_timeout_ms", 1000, config::AtLeast(1));
  options.drain_timeout_ms = reader.Get("drain_timeout_ms", 2000, config::AtLeast(1));
  options.batch_size = reader.Get("batch_size", 64, config::AtLeast(1));
  options.batch_bytes = reader.Get("batch_bytes", 8 << 20, config::AtLeast(1));
  return options;
}

std::unique_ptr<Publisher> MakeRedisStreamPublisher(RedisOptions options) {
  return std::make_unique<RedisPublisher>(std::move(options));
}

}  // namespace dsf::publish
