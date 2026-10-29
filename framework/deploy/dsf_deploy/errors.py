"""部署工具的错误类型。"""


class DeployError(Exception):
    """部署工具的预期失败：消息直接给用户看，不打印堆栈。"""
