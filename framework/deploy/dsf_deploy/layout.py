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
GENERATED = "generated"
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
