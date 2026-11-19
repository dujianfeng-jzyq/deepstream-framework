"""类别列表的差异（docs/training.md §15.1）。

下游（模型、服务）记录确认时上游的类别列表，顺序即类别号。列表不同时按名称与位置报告差异，人据此判断
下游是否需要修改：训练工具的 sources.lock.json 与部署工具的 models.lock.json 用同一套说法。
"""

from __future__ import annotations


def diff(old: list[str], new: list[str]) -> list[str]:
    """old → new 的差异，每项一行；列表相同时为空。同一位置一删一增写成“第 n 类由 A 改为 B”（多半是改名），
    其余按名称报告新增、删除与编号变化。"""
    old_index = {name: index for index, name in enumerate(old)}
    new_index = {name: index for index, name in enumerate(new)}
    removed = {old_index[name]: name for name in old if name not in new_index}
    added = {new_index[name]: name for name in new if name not in old_index}
    lines = []
    for index in sorted(set(removed) & set(added)):
        lines.append(f"class {index}: {removed.pop(index)} -> {added.pop(index)}")
    lines += [f"added {name} (class {index})" for index, name in sorted(added.items())]
    lines += [f"removed {name} (was class {index})" for index, name in sorted(removed.items())]
    moved = [f"{name} {old_index[name]}->{index}" for index, name in enumerate(new)
             if name in old_index and old_index[name] != index]
    if moved:
        lines.append("renumbered " + ", ".join(moved))
    return lines
