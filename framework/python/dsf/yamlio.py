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


def load(source):
    """读取文本或文件对象；空文档返回 None。"""
    return yaml.load(source, Loader=StrictLoader)


def dump(document) -> str:
    """保持键顺序、输出 UTF-8 原文。"""
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
