// 极简测试支持：TEST 注册、CHECK 断言、SKIP。不引入第三方测试框架，测试可执行文件
// 直接由 CTest 运行；返回码非零即失败。

#ifndef DSF_TEST_H
#define DSF_TEST_H

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dsf_test {

struct Case {
  const char* name;
  std::function<void()> body;
};

inline std::vector<Case>& Registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Register {
  Register(const char* name, std::function<void()> body) { Registry().push_back({name, std::move(body)}); }
};

struct Failure : std::exception {
  explicit Failure(std::string text) : message(std::move(text)) {}
  const char* what() const noexcept override { return message.c_str(); }
  std::string message;
};

struct Skip : std::exception {
  explicit Skip(std::string text) : message(std::move(text)) {}
  const char* what() const noexcept override { return message.c_str(); }
  std::string message;
};

inline int RunAll() {
  int failed = 0;
  for (const Case& test : Registry()) {
    try {
      test.body();
      std::cout << "[ OK ] " << test.name << "\n";
    } catch (const Skip& skip) {
      std::cout << "[SKIP] " << test.name << ": " << skip.what() << "\n";
    } catch (const std::exception& error) {
      std::cout << "[FAIL] " << test.name << ": " << error.what() << "\n";
      ++failed;
    }
  }
  std::cout << Registry().size() - static_cast<std::size_t>(failed) << "/" << Registry().size()
            << " passed\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace dsf_test

#define DSF_TEST_CONCAT2(a, b) a##b
#define DSF_TEST_CONCAT(a, b) DSF_TEST_CONCAT2(a, b)
#define TEST(name)                                                                       \
  static void name();                                                                    \
  static const dsf_test::Register DSF_TEST_CONCAT(register_, name)(#name, &name);        \
  static void name()

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::ostringstream where;                                                          \
      where << __FILE__ << ":" << __LINE__ << ": CHECK(" #condition ") failed";          \
      throw dsf_test::Failure(where.str());                                              \
    }                                                                                    \
  } while (false)

#define CHECK_EQ(actual, expected)                                                       \
  do {                                                                                   \
    const auto& actual_value = (actual);                                                 \
    const auto& expected_value = (expected);                                             \
    if (!(actual_value == expected_value)) {                                             \
      std::ostringstream where;                                                          \
      where << __FILE__ << ":" << __LINE__ << ": " #actual " == " #expected              \
            << "\n    actual:   " << actual_value << "\n    expected: " << expected_value; \
      throw dsf_test::Failure(where.str());                                              \
    }                                                                                    \
  } while (false)

// 断言表达式抛出异常，且消息包含 fragment。
#define CHECK_THROWS(expression, fragment)                                               \
  do {                                                                                   \
    bool thrown = false;                                                                 \
    try {                                                                                \
      (void)(expression);                                                                \
    } catch (const std::exception& error) {                                              \
      thrown = true;                                                                     \
      if (std::string(error.what()).find(fragment) == std::string::npos) {               \
        std::ostringstream where;                                                        \
        where << __FILE__ << ":" << __LINE__ << ": exception '" << error.what()          \
              << "' does not mention '" << fragment << "'";                              \
        throw dsf_test::Failure(where.str());                                            \
      }                                                                                  \
    }                                                                                    \
    if (!thrown) {                                                                       \
      std::ostringstream where;                                                          \
      where << __FILE__ << ":" << __LINE__ << ": " #expression " did not throw";         \
      throw dsf_test::Failure(where.str());                                              \
    }                                                                                    \
  } while (false)

#define SKIP(reason) throw dsf_test::Skip(reason)

#define DSF_TEST_MAIN() \
  int main() { return dsf_test::RunAll(); }

#endif  // DSF_TEST_H
