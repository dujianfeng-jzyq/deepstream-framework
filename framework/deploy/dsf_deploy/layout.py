"""项目与发布包的文件布局，以及容器内的固定挂载点。全部路径约定只在这里定义。"""

PROJECT_ENV = "project.env"
SITE_FIELDS = "site-fields.env"
SITE_ENV = "site.env"
ENV = ".env"
RELEASE_ENV = "release.env"
INVENTORY = "package-files.json"
MANAGED = ".managed-files.json"
COMPOSE_ROOT = "docker-compose.yaml"
PROFILES = "profiles"
SERVICES = "services"
PLUGINS = "plugins"
GENERATED = "generated"
# profile 设置 x-network-mode 时生成的 Compose 覆盖文件，排在 COMPOSE_FILE 最后。
NETWORK_COMPOSE = "generated/network.compose.yaml"
MODELS = "models"
UNIT_COMPOSE = "compose.yaml"
UNIT_TEMPLATES = "templates"

# 容器内挂载点：所有服务一致，应用通过同名环境变量读取。
MOUNTS = {
    "CONTAINER_WORKSPACE": "/workspace",
    "CONTAINER_CONFIG_PATH": "/config",
    "CONTAINER_DATA_PATH": "/data",
    "CONTAINER_MODELS_PATH": "/models",
}

# 发布包中的部署文件（目录按树复制）；服务单元只带 compose.yaml 与 templates/。
PACKAGE_TREES = ("framework/compose", "framework/deploy", "framework/python/dsf", PROFILES, MODELS)
PACKAGE_FILES = (COMPOSE_ROOT, SITE_FIELDS)
# 设备相关的派生物不进包：engine 必须在目标机构建。
PACKAGE_EXCLUDED_SUFFIXES = (".engine", ".fingerprint", ".lock", ".pyc")
PACKAGE_EXCLUDED_NAMES = ("__pycache__", ".DS_Store")

# 服务声明支持无副作用预检的标签。
CONFIG_CHECK_LABEL = "io.deepstream.config-check"
PAYLOAD_VOLUME = "app-files"
# apps 镜像记录构建时源码版本的标签（naming.source_revision），打包时比对。
SOURCE_REVISION_LABEL = "io.deepstream.source-revision"


def payload_volume(project_name: str) -> str:
    """Compose 为载荷卷生成的全名（项目名来自 docker-compose.yaml 的 name: ${PROJECT_NAME}）。"""
    return f"{project_name}_{PAYLOAD_VOLUME}"
