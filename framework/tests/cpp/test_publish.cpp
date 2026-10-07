// dsf::publish：Redis Stream 发布器。需要本机 redis-server（缺失时 SKIP）。

#include <fcntl.h>
#include <hiredis/hiredis.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <thread>

#include "dsf/publish/publisher.h"
#include "dsf_test.h"

extern char** environ;

using namespace dsf::publish;

namespace {

std::string Which(const char* program) {
  std::string command = std::string("command -v ") + program + " 2>/dev/null";
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) return {};
  char buffer[512] = {};
  std::string path = fgets(buffer, sizeof(buffer), pipe) != nullptr ? buffer : "";
  pclose(pipe);
  while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) path.pop_back();
  return path;
}

int FreePort() {
  static int next = 20000 + static_cast<int>(getpid() % 20000);
  return next++;
}

class RedisServer {
 public:
  explicit RedisServer(const std::string& password = {}) : port_(FreePort()) {
    const std::string binary = Which("redis-server");
    if (binary.empty()) SKIP("redis-server not installed");
    std::vector<std::string> args = {binary, "--port", std::to_string(port_), "--save", "",
                                     "--appendonly", "no", "--bind", "127.0.0.1"};
    if (!password.empty()) {
      args.push_back("--requirepass");
      args.push_back(password);
    }
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    // 服务端日志与测试输出无关，丢弃。
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    const int spawned = posix_spawn(&pid_, binary.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) SKIP("cannot start redis-server");
    for (int i = 0; i < 100; ++i) {
      redisContext* context = redisConnect("127.0.0.1", port_);
      const bool ok = context != nullptr && context->err == 0;
      if (context != nullptr) redisFree(context);
      if (ok) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw dsf_test::Failure("redis-server did not start");
  }
  ~RedisServer() {
    kill(pid_, SIGTERM);
    waitpid(pid_, nullptr, 0);
  }
  int port() const { return port_; }

  // 最后一条记录的字段。
  Fields Last(const std::string& stream) const {
    redisContext* context = redisConnect("127.0.0.1", port_);
    auto* reply = static_cast<redisReply*>(redisCommand(context, "XREVRANGE %s + - COUNT 1", stream.c_str()));
    Fields fields;
    if (reply != nullptr && reply->type == REDIS_REPLY_ARRAY && reply->elements == 1) {
      const redisReply* values = reply->element[0]->element[1];
      for (std::size_t i = 0; i + 1 < values->elements; i += 2) {
        fields.emplace_back(std::string(values->element[i]->str, values->element[i]->len),
                            std::string(values->element[i + 1]->str, values->element[i + 1]->len));
      }
    }
    if (reply != nullptr) freeReplyObject(reply);
    redisFree(context);
    return fields;
  }

  long long Length(const std::string& stream, const std::string& password = {}) const {
    redisContext* context = redisConnect("127.0.0.1", port_);
    if (!password.empty()) freeReplyObject(redisCommand(context, "AUTH %s", password.c_str()));
    auto* reply = static_cast<redisReply*>(redisCommand(context, "XLEN %s", stream.c_str()));
    const long long length = reply != nullptr ? reply->integer : -1;
    if (reply != nullptr) freeReplyObject(reply);
    redisFree(context);
    return length;
  }

 private:
  int port_;
  pid_t pid_ = 0;
};

RedisOptions Options(int port) {
  RedisOptions options;
  options.port = port;
  options.drain_timeout_ms = 2000;
  return options;
}

}  // namespace

TEST(publishes_to_independent_channels) {
  RedisServer server;
  auto publisher = MakeRedisStreamPublisher(Options(server.port()));
  Channel& detections = publisher->Open({"test:detections", 1000, 1000});
  Channel& recordings = publisher->Open({"test:recordings", 10, 1000});
  CHECK(!detections.TryPublish("early"));  // 未启动时丢弃
  publisher->Start();
  for (int i = 0; i < 200; ++i) CHECK(detections.TryPublish("{\"n\":" + std::to_string(i) + "}"));
  CHECK(recordings.TryPublish("{\"event\":1}"));
  publisher->Stop();
  CHECK_EQ(detections.Stats().published, 200u);
  CHECK_EQ(detections.Stats().dropped, 1u);
  CHECK_EQ(recordings.Stats().published, 1u);
  CHECK_EQ(server.Length("test:detections"), 200LL);
  CHECK_EQ(server.Length("test:recordings"), 1LL);
  CHECK(!detections.TryPublish("late"));  // 停止后丢弃
  CHECK_THROWS(publisher->Open({"x", 1, 1}), "before Start");
}

TEST(publishes_binary_fields_and_trims_to_max_length) {
  RedisServer server;
  RedisOptions options = Options(server.port());
  options.batch_bytes = 1;  // 每轮只发一条：字节上限不能让大消息永远发不出去
  auto publisher = MakeRedisStreamPublisher(options);
  Channel& channel = publisher->Open({"test:images", 300, 10});
  publisher->Start();
  const std::string jpeg("\xFF\xD8\x00\r\n\xFF\xD9", 7);
  for (int i = 0; i < 250; ++i) CHECK(channel.TryPublish(Fields{{"payload", "{}"}, {"image", jpeg}}));
  CHECK(!channel.TryPublish(Fields{}));
  publisher->Stop();
  CHECK_EQ(channel.Stats().published, 250u);
  CHECK_EQ(channel.Stats().dropped, 1u);
  const Fields last = server.Last("test:images");
  CHECK_EQ(last.size(), 2u);
  CHECK_EQ(last[1].first, std::string("image"));
  CHECK(last[1].second == jpeg);
  // MAXLEN ~ 按整个内部节点（默认 100 条）裁剪：长度不超过上限加一个节点。
  CHECK(server.Length("test:images") <= 10LL + 100LL);
}

TEST(byte_limit_drops_when_backlog_is_full) {
  RedisOptions options = Options(FreePort());  // 没有服务监听：消息只能积压
  options.connect_timeout_ms = 100;
  options.drain_timeout_ms = 200;
  auto publisher = MakeRedisStreamPublisher(options);
  ChannelSpec spec{"test:x", 100, 100};
  spec.max_bytes = 2500;
  Channel& channel = publisher->Open(spec);
  publisher->Start();
  const std::string big(1000, 'x');
  CHECK(channel.TryPublish(Fields{{"image", std::string(5000, 'x')}}));  // 空队列总能放下一条
  CHECK(!channel.TryPublish(Fields{{"image", big}}));
  publisher->Stop();
  CHECK_EQ(channel.Stats().dropped, 2u);  // 1 条超限 + 1 条排空期满仍未发出
}

TEST(buffers_while_disconnected_and_drops_on_overflow) {
  RedisOptions options = Options(FreePort());  // 没有服务监听
  options.connect_timeout_ms = 100;
  options.drain_timeout_ms = 200;
  auto publisher = MakeRedisStreamPublisher(options);
  Channel& channel = publisher->Open({"test:x", 3, 100});
  publisher->Start();
  int accepted = 0;
  for (int i = 0; i < 10; ++i) accepted += channel.TryPublish("m") ? 1 : 0;
  CHECK_EQ(accepted, 3);
  publisher->Stop();
  const ChannelStats stats = channel.Stats();
  CHECK_EQ(stats.published, 0u);
  CHECK_EQ(stats.dropped, 10u);  // 7 条溢出 + 3 条排空期满仍未发出
  CHECK(!publisher->connected());
}

TEST(authenticates_with_password) {
  RedisServer server("s3cret");
  RedisOptions wrong = Options(server.port());
  wrong.password = "nope";
  wrong.drain_timeout_ms = 300;
  auto rejected = MakeRedisStreamPublisher(wrong);
  Channel& lost = rejected->Open({"auth:x", 10, 100});
  rejected->Start();
  lost.TryPublish("m");
  rejected->Stop();
  CHECK_EQ(lost.Stats().published, 0u);

  RedisOptions right = Options(server.port());
  right.password = "s3cret";
  auto accepted = MakeRedisStreamPublisher(right);
  Channel& channel = accepted->Open({"auth:x", 10, 100});
  accepted->Start();
  channel.TryPublish("m");
  accepted->Stop();
  CHECK_EQ(channel.Stats().published, 1u);
  CHECK_EQ(server.Length("auth:x", "s3cret"), 1LL);
}

TEST(redis_options_read_from_config) {
  const auto document = dsf::config::Document::LoadText("redis:\n  host: r\n  port: 7000\n  password: p\n", "t");
  dsf::config::Reader root = document.Root();
  const RedisOptions options = RedisOptions::Read(root.Section("redis"));
  root.Finish();
  CHECK_EQ(options.host, std::string("r"));
  CHECK_EQ(options.port, 7000);
  CHECK(root.Describe().find("redis.password=***") != std::string::npos);
  const auto bad = dsf::config::Document::LoadText("redis:\n  port: 0\n", "t");
  dsf::config::Reader bad_root = bad.Root();
  CHECK_THROWS(RedisOptions::Read(bad_root.Section("redis")), "redis.port: must be between 1 and 65535");
}

DSF_TEST_MAIN()
