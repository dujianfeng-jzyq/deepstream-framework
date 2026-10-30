"""现场参数：清单（site-fields.env）声明要问什么，答案（site.env）记录现场的回答。

清单语法：
    # 紧邻键上方的注释就是提问提示（空行断开），每行注释显示为一行
    KEY=          必填
    KEY=""        可以为空
    KEY=默认值    可选，回车即用默认值
键名含 PASSWORD/PASSWD/TOKEN/SECRET 的输入不回显、日志与回显中打码。
答案文件中的值可以加一对引号（KEY="值"），读取时去掉；引号之内按字面。
现场键不能占用项目身份、版本、镜像与挂载点等由项目或渲染器决定的键。
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import envfile
from .errors import DeployError

SECRET_MARKERS = ("PASSWORD", "PASSWD", "TOKEN", "SECRET")
CLEAR = "CLEAR"  # 提问时输入它把可留空的字段清空
RESERVED_KEYS = {"PROJECT_NAME", "CONFIG_PROFILE", "HOST_UID", "HOST_GID", "HOST_DATA_PATH",
                 "INSTALL_ROOT", "PIP_INDEX_URL", "BASE_APT_PACKAGES", "TARGET_PLATFORM",
                 "DEEPSTREAM_VERSION", "BASE_INPUTS", "SOURCE_REVISION"}
RESERVED_PREFIXES = ("CONTAINER_", "COMPOSE_", "DSF_")
RESERVED_SUFFIXES = ("_IMAGE", "_VERSION")


@dataclass(frozen=True)
class Field:
    key: str
    hint: tuple[str, ...]
    default: str
    nullable: bool

    @property
    def prompt(self) -> str:
        """一行的说明，用于错误信息。"""
        return " ".join(self.hint) or self.key

    @property
    def secret(self) -> bool:
        return any(marker in self.key.upper() for marker in SECRET_MARKERS)

    @property
    def required(self) -> bool:
        return not self.default and not self.nullable


def is_reserved(key: str) -> bool:
    return (key in RESERVED_KEYS or key.startswith(RESERVED_PREFIXES) or key.endswith(RESERVED_SUFFIXES))


def parse_fields(path: Path) -> list[Field]:
    if not path.is_file():
        return []
    fields: list[Field] = []
    comments: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            comments.clear()
            continue
        if line.startswith("#"):
            comments.append(line.lstrip("#").strip())
            continue
        if "=" not in line:
            raise DeployError(f"{path}:{number}：应为 KEY=VALUE，实际是 {raw!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or is_reserved(key):
            raise DeployError(f"{path}:{number}：{key!r} 不能作为现场参数（保留键或空键）")
        if any(field.key == key for field in fields):
            raise DeployError(f"{path}:{number}：现场参数 {key} 重复")
        nullable = value in ('""', "''")
        fields.append(Field(key, tuple(c for c in comments if c), "" if nullable else value, nullable))
        comments.clear()
    return fields


def read_answers(path: Path, fields: list[Field], retired: list[str] | None = None) -> dict[str, str]:
    """读取现场答案；只能包含清单声明的键。传入 retired 时（升级：新版本可能删除了字段）
    未声明的键不报错，从答案中去掉并记入 retired。"""
    if not path.is_file():
        return {}
    answers = envfile.read(path, unquote=envfile.unquote_answer)
    unknown = sorted(set(answers) - {field.key for field in fields})
    if unknown and retired is None:
        raise DeployError(f"{path}：未声明的现场参数 {unknown}；请在 site-fields.env 中声明")
    if unknown:
        retired.extend(unknown)
    return {key: value for key, value in answers.items() if key not in unknown}


def write_answers(path: Path, answers: dict[str, str], header: tuple[str, ...] = (),
                  mode: int | None = None) -> None:
    """写成 KEY="值"：现场照此格式手工编辑时，KEY="" 表示空值（与清单语法一致）。"""
    envfile.write(path, answers, header, mode, envfile.quote_answer)


def validate(fields: list[Field], values: dict[str, str]) -> None:
    for field in fields:
        if field.key in values and not values[field.key] and not field.nullable:
            raise DeployError(f"{field.key} 不能为空（{field.prompt}）；"
                              "请在 site.env 中填写，或删除这个键以使用默认值")


def resolve(fields: list[Field], answers: dict[str, str], inherited: dict[str, str]) -> dict[str, str]:
    """优先级：清单默认值 < 继承值（project.env、profile）< 现场答案；显式空答案不回退。"""
    values = defaults(fields, inherited)
    values.update(answers)
    validate(fields, answers)
    missing = [field.key for field in fields if field.required and not values.get(field.key)]
    if missing:
        raise DeployError(f"必填的现场参数没有填写：{', '.join(missing)}")
    return values


def ask(field: Field, default: str | None = None, prompt: Callable[[str], str] = input,
        prompt_secret: Callable[[str], str] = getpass.getpass, position: str = "") -> str:
    """default 是回车接受的值：首次提问时是继承后的默认值（见 resolve），它会作为答案保存，
    所以必须与渲染时的继承值一致，否则会遮住 project.env 或 profile 的设置；重新提问（--ask）时
    是已保存的答案。可留空的字段有默认值时输入 CLEAR 清空（回车只能接受默认值）。
    提示分行：标题行是序号、键名与回车的效果，其下是清单中的注释，最后一行输入。"""
    default = field.default if default is None else default
    reader = prompt_secret if field.secret else prompt
    status = f" [{'***' if field.secret else default}]" if default else (
        "（可留空）" if field.nullable else "（必填）")
    if default and field.nullable:
        status += f"（输入 {CLEAR} 清空）"
    title = f"{position} {field.key}{status}".lstrip()
    text = "\n".join([f"\n{title}", *(f"  {line}" for line in field.hint), "> "])
    while True:
        answer = reader(text).strip()
        if answer == CLEAR and field.nullable:
            return ""
        if answer or default or field.nullable:
            return answer or default
        print(f"  {field.key} 是必填项")
        text = "> "


def defaults(fields: list[Field], inherited: dict[str, str]) -> dict[str, str]:
    """继承后的默认值：清单默认值 < 继承值（project.env、profile）；渲染与交互提问共用。"""
    return {field.key: inherited.get(field.key, field.default) for field in fields}


def masked(fields: list[Field], values: dict[str, str]) -> list[str]:
    """每个字段一行 KEY  值，键名对齐；口令打码，空值显示为（空）。"""
    shown = [f for f in fields if f.key in values]
    width = max((len(f.key) for f in shown), default=0)
    return [f"{f.key:<{width}}  {'***' if f.secret else values[f.key]}" if values[f.key]
            else f"{f.key:<{width}}  （空）" for f in shown]
