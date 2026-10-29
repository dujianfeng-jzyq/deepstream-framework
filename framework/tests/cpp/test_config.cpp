// dsf::config 的读取、约束、未知键与摘要。

#include "dsf/config/config.h"
#include "dsf_test.h"

using dsf::config::Document;
using dsf::config::Reader;

namespace {

Document Load(const std::string& text) { return Document::LoadText(text, "test.yaml"); }

}  // namespace

TEST(reads_typed_values_and_defaults) {
  const Document document = Load("a:\n  n: 3\n  d: 1.5\n  b: false\n  s: hello\n");
  Reader root = document.Root();
  Reader a = root.Section("a");
  CHECK_EQ(a.Get("n", 0), 3);
  CHECK_EQ(a.Get("d", 0.0), 1.5);
  CHECK_EQ(a.Get("b", true), false);
  CHECK_EQ(a.Get<std::string>("s", ""), std::string("hello"));
  CHECK_EQ(a.Get("missing", 7), 7);
  root.Finish();
}

TEST(null_values_use_defaults) {
  const Document document = Load("a:\n  n:\nb:\n");
  Reader root = document.Root();
  CHECK_EQ(root.Section("a").Get("n", 5), 5);
  CHECK_EQ(root.Section("b").Get("x", 1), 1);
  root.Finish();
}

TEST(required_and_constraints_report_full_path) {
  const Document document = Load("a:\n  n: -1\n");
  Reader root = document.Root();
  CHECK_THROWS(root.Section("a").Get("n", 0, dsf::config::AtLeast(0)), "test.yaml: a.n: must be at least 0");
  CHECK_THROWS(root.Required<std::string>("id"), "id: is required");
}

TEST(warnings_are_collected_with_full_path) {
  const Document document = Load("a:\n  n: 3\n");
  Reader root = document.Root();
  Reader a = root.Section("a");
  CHECK_EQ(a.Get("n", 0), 3);
  CHECK(root.Warnings().empty());
  a.Warn("n", "is unusual");
  root.Warn("", "check the outputs");
  root.Finish();  // 警告不影响校验
  const auto warnings = root.Warnings();
  CHECK_EQ(warnings.size(), std::size_t{2});
  CHECK_EQ(warnings[0], std::string("test.yaml: a.n: is unusual"));
  CHECK_EQ(warnings[1], std::string("test.yaml: (top level): check the outputs"));
}

TEST(type_errors_are_rejected) {
  const Document document = Load("n: abc\nq: \"12\"\nb: maybe\nd: .nan\nl: [1]\n");
  Reader root = document.Root();
  CHECK_THROWS(root.Get("n", 0), "expected an integer");
  CHECK_THROWS(root.Get("q", 0), "expected an integer");  // 引号标量不是数字
  CHECK_EQ(root.Get<std::string>("q", ""), std::string("12"));
  CHECK_THROWS(root.Get("b", false), "expected a boolean");
  CHECK_THROWS(root.Get("d", 0.0), "finite");
  CHECK_THROWS(root.Get<std::string>("l", ""), "got a list");
}

TEST(unknown_keys_are_rejected_with_allowed_list) {
  const Document document = Load("a:\n  known: 1\n  typo: 2\n");
  Reader root = document.Root();
  (void)root.Section("a").Get("known", 0);
  CHECK_THROWS(root.Finish(), "a.typo: unknown field; allowed here: known");
}

TEST(unknown_empty_sections_are_rejected) {
  const Document document = Load("extra:\n");
  CHECK_THROWS(document.Root().Finish(), "extra: unknown field");
}

TEST(duplicate_and_dotted_keys_are_rejected) {
  CHECK_THROWS(Load("a: 1\na: 2\n"), "duplicate key");
  CHECK_THROWS(Load("a.b: 1\n"), "nested mappings");
  CHECK_THROWS(Load("- 1\n"), "top level must be a mapping");
}

TEST(lists_of_mappings) {
  const Document document = Load("items:\n  - id: x\n  - id: y\n    bad: 1\n");
  Reader root = document.Root();
  std::vector<std::string> ids;
  for (Reader item : root.List("items", 1)) ids.push_back(item.Required<std::string>("id"));
  CHECK_EQ(ids.size(), 2u);
  CHECK_EQ(ids[1], std::string("y"));
  CHECK_THROWS(root.Finish(), "items[1].bad: unknown field; allowed here: id");
  const Document empty = Load("items: []\n");
  Reader empty_root = empty.Root();
  CHECK_THROWS(empty_root.List("items", 1), "at least 1");
}

TEST(properties_pass_through_as_text) {
  const Document document = Load("p:\n  bitrate: 4000\n  preset: fast\n");
  Reader root = document.Root();
  const auto properties = root.Props("p");
  CHECK_EQ(properties.size(), 2u);
  CHECK_EQ(properties[0].name, std::string("bitrate"));
  CHECK_EQ(properties[0].value, std::string("4000"));
  CHECK_EQ(properties[1].origin, std::string("p.preset"));
  root.Finish();
  const Document nested = Load("p:\n  x: {a: 1}\n");
  Reader nested_root = nested.Root();
  CHECK_THROWS(nested_root.Props("p"), "p.x: property values must be scalars");
}

TEST(describe_masks_secrets) {
  const Document document = Load("user: alice\npassword: s3cret\n");
  Reader root = document.Root();
  (void)root.Get<std::string>("user", "");
  (void)root.Secret("password");
  const std::string text = root.Describe();
  CHECK(text.find("user=alice") != std::string::npos);
  CHECK(text.find("password=***") != std::string::npos);
  CHECK(text.find("s3cret") == std::string::npos);
}

TEST(declared_paths_cover_template_leaves) {
  const Document document = Load("a:\n  x: 1\nitems:\n  - id: q\n");
  Reader root = document.Root();
  (void)root.Section("a").Get("x", 0);
  (void)root.Section("a").Get("y", 0);
  for (Reader item : root.List("items")) (void)item.Required<std::string>("id");
  const auto leaves = document.LeafPaths();
  const auto declared = root.DeclaredPaths();
  for (const auto& leaf : leaves) {
    CHECK(std::find(declared.begin(), declared.end(), leaf) != declared.end());
  }
}

TEST(path_component_and_one_of) {
  const Document document = Load("id: a/b\nmode: x\n");
  Reader root = document.Root();
  CHECK_THROWS(root.Get<std::string>("id", "", dsf::config::PathComponent()), "single path component");
  CHECK_THROWS(root.Get<std::string>("mode", "", dsf::config::OneOf({"y", "z"})), "not one of: y z");
}

DSF_TEST_MAIN()
