"""Docker 与 Compose 的唯一调用入口。测试用替身实现同名方法即可，不需要 monkeypatch 子进程。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .envfile import DeployError

# 安装时 Compose 文件只由安装目录的 .env 决定，调用 shell 中的这些变量会让预检与执行看到不同的定义。
COMPOSE_SELECTORS = ("COMPOSE_FILE", "COMPOSE_ENV_FILES", "COMPOSE_DISABLE_ENV_FILE",
                     "COMPOSE_PATH_SEPARATOR", "COMPOSE_PROJECT_NAME", "COMPOSE_PROFILES")


def clean_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in COMPOSE_SELECTORS}


class Docker:
    def available(self) -> bool:
        return shutil.which("docker") is not None

    def _run(self, argv: list[str], capture: bool = False, cwd: Path | None = None,
             env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None)
        if result.returncode != 0:
            detail = (result.stderr or "").strip() if capture else ""
            raise DeployError(f"command failed ({result.returncode}): {' '.join(argv)}"
                              + (f"\n{detail}" if detail else ""))
        return result

    def _ok(self, argv: list[str]) -> bool:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    # ---- 镜像 ----

    def image_exists(self, image: str) -> bool:
        return self._ok(["docker", "image", "inspect", image])

    def image_platform(self, image: str) -> str:
        """镜像的架构（amd64/arm64）；镜像不存在返回空串。"""
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Architecture}}", image],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    def image_label(self, image: str, key: str) -> str:
        """镜像标签的值；镜像或标签不存在返回空串。"""
        result = subprocess.run(["docker", "image", "inspect", "--format",
                                 f'{{{{index .Config.Labels "{key}"}}}}', image], capture_output=True, text=True)
        value = result.stdout.strip() if result.returncode == 0 else ""
        return "" if value == "<no value>" else value

    def pull(self, image: str, platform: str) -> None:
        self._run(["docker", "pull", "--platform", f"linux/{platform}", image])

    def save(self, image: str, output: Path) -> None:
        self._run(["docker", "save", "-o", str(output), image])

    def load(self, archive: Path) -> None:
        self._run(["docker", "load", "-i", str(archive)])

    def volume_exists(self, name: str) -> bool:
        return self._ok(["docker", "volume", "inspect", name])

    def volume_remove(self, name: str) -> None:
        self._run(["docker", "volume", "rm", name], capture=True)

    def export_directory(self, image: str, directory: str, target: Path) -> None:
        """把镜像中的目录复制出来（创建不运行的临时容器）。"""
        container = self._run(["docker", "create", image], capture=True).stdout.strip()
        try:
            self._run(["docker", "cp", f"{container}:{directory}/.", str(target)])
        finally:
            self._run(["docker", "rm", container], capture=True)

    def run(self, args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
        return self._run(["docker", "run", *args], capture=capture)

    # ---- Compose ----

    def compose(self, project_dir: Path, args: list[str], capture: bool = False,
                env_file: Path | None = None, files: list[Path] | None = None) -> subprocess.CompletedProcess:
        argv = ["docker", "compose", "--project-directory", str(project_dir)]
        if env_file is not None:
            argv += ["--env-file", str(env_file)]
        for file in files or []:
            argv += ["-f", str(file)]
        return self._run(argv + args, capture=capture, cwd=project_dir, env=clean_environment())

    def compose_config(self, project_dir: Path, env_file: Path | None = None,
                       files: list[Path] | None = None) -> dict:
        result = self.compose(project_dir, ["config", "--format", "json"], capture=True,
                              env_file=env_file, files=files)
        return json.loads(result.stdout)

    def compose_ps(self, project_dir: Path) -> list[dict]:
        text = self.compose(project_dir, ["ps", "--all", "--format", "json"], capture=True).stdout.strip()
        if not text:
            return []
        # Compose 2.21 之前输出 JSON 数组，之后每行一个对象。
        return json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines()]
