"""从框架模板生成一个单元到指定目录（与 scripts/new-app.sh 同一实现），供 CMake 在配置阶段生成模板单元。

    generate_template_unit.py <kind> <name> <target>
"""

import shutil
import sys
from pathlib import Path

from dsf_deploy.errors import DeployError
from dsf_deploy.project import generate_unit


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    kind, name, target = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    shutil.rmtree(target, ignore_errors=True)  # 模板删除的文件不能残留
    try:
        generate_unit(kind, name, target)
    except DeployError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
