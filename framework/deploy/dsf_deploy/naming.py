"""镜像、发布包与平台的命名规则。名称只在这里派生，Compose 与安装器只消费结果。"""

from __future__ import annotations

import hashlib
import platform
import re

from .envfile import DeployError

_VERSION = re.compile(r"^\d+(\.\d+)*$")
_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
# 镜像与发布包的平台名（Docker 的 architecture）。
PLATFORMS = ("amd64", "arm64")


def deepstream_version(image: str) -> str:
    """从上游镜像 tag 的第一段取 DeepStream 版本（8.0-samples → 8.0），解析不出返回空串。"""
    reference = image.split("@", 1)[0]
    name, colon, tag = reference.rpartition(":")
    if not colon or not name or "/" in tag:
        return ""
    first = tag.split("-", 1)[0]
    return first if _VERSION.match(first) else ""


def base_inputs_digest(dockerfile_text: str, deepstream_image: str, apt_packages: str) -> str:
    """base 镜像构建输入的摘要，写进 base 的 tag：输入变了 tag 一定变，同名 tag 不会代表两种内容。"""
    digest = hashlib.sha256()
    for part in (dockerfile_text, deepstream_image, " ".join(apt_packages.split())):
        digest.update(part.encode("utf-8") + b"\0")
    return digest.hexdigest()[:8]


def image_names(project: str, ds_version: str, apps_version: str, base_version: str,
                base_digest: str) -> dict[str, str]:
    segment = f"-{ds_version}" if ds_version else ""
    return {
        "BASE_IMAGE": f"{project}/deepstream{segment}-base:{base_version}-{base_digest}",
        "APPS_IMAGE": f"{project}/deepstream{segment}-apps:{apps_version}",
    }


def package_name(project: str, apps_version: str, ds_version: str) -> str:
    return f"{project}-v{apps_version}" + (f"-ds{ds_version}" if ds_version else "")


def image_tar_name(image: str) -> str:
    """包内 tar 文件名：保留可读的名称与 tag，替换文件系统不安全的字符。"""
    reference = image.split("@", 1)[0]
    name, colon, tag = reference.rpartition(":")
    if not colon or "/" in tag:  # 没有 tag，或冒号属于 registry 端口
        name, tag = reference, ""
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", name).strip("-")
    return f"{safe}-{tag}.tar" if tag else f"{safe}.tar"


def docker_arch(machine: str) -> str:
    value = machine.lower()
    if value not in _ARCH:
        raise DeployError(f"unsupported machine architecture: {machine!r}")
    return _ARCH[value]


def host_platform() -> str:
    return docker_arch(platform.machine())
