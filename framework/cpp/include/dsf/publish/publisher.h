// 消息发布：媒体线程只入队，网络 IO 在后台线程。
//
// 语义（默认实现 RedisStreamPublisher）：内存队列、至多一次。队列满、未启动或已停止时丢弃
// 并计数；不落盘、不在断线或重启后补发。需要可靠投递时实现自己的 Publisher，
// 或在图内使用 nvmsgbroker。

#ifndef DSF_PUBLISH_PUBLISHER_H
#define DSF_PUBLISH_PUBLISHER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dsf/config/config.h"

namespace dsf::publish {

struct ChannelStats {
  std::uint64_t published = 0;  // 后端已确认接收
  std::uint64_t dropped = 0;    // 队列满、未运行或停止时仍未发出
  std::uint64_t failed = 0;     // 后端拒绝或发送中断
  std::uint64_t queued = 0;
};

struct ChannelSpec {
  std::string target;      // Redis：Stream 键名
  int capacity = 256;      // 队列容量：后端慢或断线时能缓冲的消息数
  int max_length = 10000;  // Redis：XADD MAXLEN ~ 近似上限
};

class Channel {
 public:
  virtual ~Channel() = default;
  // 不阻塞；返回 false 表示消息已被丢弃（并计数）。可在任意线程调用。
  virtual bool TryPublish(std::string payload) = 0;
  virtual ChannelStats Stats() const = 0;
  virtual const std::string& target() const = 0;
};

class Publisher {
 public:
  virtual ~Publisher() = default;
  // 只能在 Start 之前打开通道；返回的引用在 Publisher 存活期间有效。
  virtual Channel& Open(const ChannelSpec& spec) = 0;
  // 一次性：Stop 之后不能再次 Start。
  virtual void Start() = 0;
  // 有界排空后停止；幂等。
  virtual void Stop() = 0;
  virtual bool connected() const = 0;
  virtual std::vector<const Channel*> Channels() const = 0;
  virtual std::string Describe() const = 0;
};

struct RedisOptions {
  std::string host = "127.0.0.1";
  int port = 6379;
  std::string password;
  int connect_timeout_ms = 1000;
  int command_timeout_ms = 1000;  // 半死的连接不能让后台线程一直卡住
  int drain_timeout_ms = 2000;    // 停止时为剩余消息保留的时间
  int batch_size = 64;            // 每轮 pipelining 发送的最大条数

  // 读取 host/port/password/*_timeout_ms/batch_size。
  static RedisOptions Read(config::Reader reader);
};

std::unique_ptr<Publisher> MakeRedisStreamPublisher(RedisOptions options);

}  // namespace dsf::publish

#endif  // DSF_PUBLISH_PUBLISHER_H
