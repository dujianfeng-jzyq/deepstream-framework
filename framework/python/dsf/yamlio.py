"""严格的 YAML 读取：拒绝重复键与非字符串键，与 C++ 配置库的规则一致。

部署工具（渲染）与 Python 服务共用；只依赖 PyYAML。
"""

import yaml


class StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, f"mapping keys must be strings, got {key!r}", key_node.start_mark)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class Plain(str):
    """未加引号的标量原文。类型由读取方决定，与 C++ 配置库（yaml-cpp）一致：
    password: 123456 读成字符串时是 "123456"，读成整数时是 123；加引号的 "123" 只能读成字符串。"""


class TextLoader(StrictLoader):
    """标量只解析 null（~、null、空值），其余保留原文：未加引号的为 Plain，加引号的为 str。
    合并键（<<）照常生效。"""


_KEPT_TAGS = ("tag:yaml.org,2002:null", "tag:yaml.org,2002:merge")
TextLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag in _KEPT_TAGS]
    for first, resolvers in StrictLoader.yaml_implicit_resolvers.items()}


def _construct_text(loader, node):
    value = loader.construct_scalar(node)
    return Plain(value) if node.style is None else value


TextLoader.add_constructor("tag:yaml.org,2002:str", _construct_text)


def load(source):
    """读取文本或文件对象；空文档返回 None。"""
    return yaml.load(source, Loader=StrictLoader)


def load_text(source):
    """同 load，但标量保持原文（见 TextLoader）；配置读取使用。"""
    return yaml.load(source, Loader=TextLoader)


class Reset(list):
    """输出为 Compose 的 !reset 标签：多文件合并时清空前面文件给出的值（Compose 2.24.4+）。"""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(Reset, lambda dumper, value: dumper.represent_sequence("!reset", value))


def dump(document) -> str:
    """保持键顺序、输出 UTF-8 原文。"""
    return yaml.dump(document, Dumper=_Dumper, sort_keys=False, allow_unicode=True)
