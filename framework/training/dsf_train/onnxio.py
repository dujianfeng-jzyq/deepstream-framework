"""读取导出的 ONNX：签名与 onnxruntime 推理。与后端无关，onnx 与 onnxruntime 只在训练镜像中。"""

from __future__ import annotations

from pathlib import Path


def signature(path: Path) -> dict:
    """{"inputs": [...], "outputs": [...]}，每项 {"name", "shape"}；动态维度记为它的名字（字符串）。
    格式与 release.json 的 onnx 相同，打包时的模型校验读它。"""
    import onnx
    graph = onnx.load(str(path), load_external_data=False).graph
    initializers = {item.name for item in graph.initializer}

    def describe(value) -> dict:
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                for d in value.type.tensor_type.shape.dim]
        return {"name": value.name, "shape": dims}
    return {"inputs": [describe(v) for v in graph.input if v.name not in initializers],
            "outputs": [describe(v) for v in graph.output]}


def run(path: Path, batch):
    """在 CPU 上推理一批输入（float32 [n, 3, 高, 宽]），返回唯一输出。CPU 结果不依赖显卡与驱动。"""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    [output] = session.get_outputs()
    return session.run([output.name], {session.get_inputs()[0].name: batch})[0]
