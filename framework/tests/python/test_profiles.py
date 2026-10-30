"""项目中的每个 profile 都能渲染，且 Compose 能解析最终定义。

依赖 profiles/ 与项目输入；builder 在编译之后才复制它们，所以本测试带 profiles 标签单独运行
（见 framework/tests/CMakeLists.txt 与 apps.Dockerfile）。
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from deploy_support import docker_cli_available
from dsf_deploy import layout, pack, render
from dsf_deploy.docker import Docker

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
class ProfilesTest(unittest.TestCase):
    def test_every_profile_renders_and_composes(self):
        profiles = [name for name, _ in render.list_profiles(ROOT)]
        self.assertTrue(profiles, "no profiles/*.yaml")
        with tempfile.TemporaryDirectory(prefix="dsf-profiles-") as temporary:
            # 与安装器相同的部署文件集合（不含模型）；不写项目根下的 .env 与 generated/。
            stage = Path(temporary)
            pack.stage_deployment_files(ROOT, stage, models=False)
            shutil.copy2(ROOT / layout.PROJECT_ENV, stage / layout.PROJECT_ENV)
            (stage / render.BASE_DOCKERFILE).parent.mkdir(parents=True)
            shutil.copy2(ROOT / render.BASE_DOCKERFILE, stage / render.BASE_DOCKERFILE)
            shutil.copy2(ROOT / layout.BUILD_COMPOSE, stage / layout.BUILD_COMPOSE)
            for name in profiles:
                with self.subTest(profile=name):
                    plan = render.build_plan(stage, name)
                    render.write_plan(stage, plan)
                    config = Docker().compose_config(stage)
                    # 请求 GPU 的服务必须在 NVIDIA runtime 下运行，否则 NVIDIA_* 变量不生效、容器内没有 GPU。
                    gpu = {service: definition.get("runtime") for service, definition in config["services"].items()
                           if "NVIDIA_VISIBLE_DEVICES" in (definition.get("environment") or {})}
                    self.assertTrue(gpu)
                    self.assertEqual(gpu, dict.fromkeys(gpu, "nvidia"))
                    if layout.NETWORK_COMPOSE in plan.files:
                        self.assertNotIn("networks", config)  # 没有服务留在默认网络


if __name__ == "__main__":
    unittest.main()
