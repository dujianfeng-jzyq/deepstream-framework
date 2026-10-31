# 模型训练与发布

本文说明从数据到现场运行的模型生命周期：准备数据 → 训练 → 评估 → 导出与发布 → 打包安装 → 在服务器或 Jetson 上运行。模型目录的格式与打包校验已经实现（§2、§7）；训练工具（§3–§6）是设计，按 §9 的顺序实现。

## 1. 原则

| 原则 | 含义 |
|---|---|
| 规格与事实分开 | `model.yaml` 是人写的规格；`release.json` 是工具写的发布事实，不手改 |
| 内容决定标识 | 数据集与发布产物用内容摘要标识；同样的输入得到同样的标识，任何改动都得到新标识 |
| 只有发布的东西进仓库 | 原始数据、数据集、训练过程与权重留在训练根目录；仓库只有规格、发布记录与部署所需的产物 |
| 契约在打包前检查 | 类别、输入尺寸、输出布局、标签与 nvinfer 配置不一致时不产出发布包，而不是到现场才发现 |
| 训练不进运行环境 | 训练镜像（PyTorch、训练框架）独立于 base/apps，不进入发布包 |
| 后端可替换 | 训练后端是适配器；规格中的 `options` 原样透传给后端，与框架块的 `properties:` 同一思路 |

## 2. 模型目录（仓库内）

```text
models/<id>/
  model.yaml          规格：类别、输入尺寸、输出布局、数据来源与划分、训练配方、发布门槛、导出参数
  release.json        发布事实：产物 sha256、导出的 ONNX 签名、来源 run 与数据集、指标（发布工具写入）
  artifacts/          已发布的产物：model.onnx、labels.txt（只能是 release.json 列出的文件）
  nvinfer.txt         nvinfer 配置：人工维护的运行参数，引用 artifacts/ 中的产物
  nvinfer.trtexec     可选：engine 构建的额外 trtexec 参数（见 deployment.md §4）
```

- `model.yaml` 的 `classes` 顺序即类别号。发布之后调换或删减类别会改变已部署业务对类别号的解读，所以规格与发布记录不一致时校验失败：要么重新训练发布，要么恢复原值。追加类别同样需要重新发布。
- `input` 与 `output_layout` 是部署契约。`output_layout` 由解析插件定义，例如 `plugins/yolo-parser` 接受 `xyxy-confidence-class_id`（唯一输出层 `[batch, 候选数, 6]`）。
- 应用配置引用 `${CONTAINER_MODELS_PATH}/<id>/nvinfer.txt`；engine 由 `model-engine-builder` 按 `model-engine-file` 在目标机构建，产物更新后指纹变化，自动重建。`model-engine-file` 必须在 `/data/engines/<id>/` 下：模型目录对所有容器只读，数据目录跨发布与清洁安装保留。
- 一个模型目录可以有多份 nvinfer 配置（例如不同精度），每份都按同样规则检查。

产物目前直接提交到仓库。模型频繁重新发布时，ONNX 会让仓库历史持续变大，可以给 `models/*/artifacts/**` 启用 Git LFS（校验与打包只看文件内容，不受影响）。

## 3. 训练根目录（仓库外）

```text
<TRAIN_ROOT>/                       默认 ~/<项目名>/training，可用 TRAIN_ROOT 指定（建议放在大容量盘）
  sources/<来源>/                   原始数据：images/ 与标注导出（YOLO txt 或 COCO json），导入后只读
  datasets/<id>/<数据集标识>/        准备好的数据集，不可变
    manifest.json                    每个样本的图像 sha256、标注、所属划分；数据集标识 = 它的摘要
    yolo/                            后端需要的视图（硬链接到 sources，不复制图像）
  runs/<id>/<run>/                  一次训练，不可变
    run.json                         规格摘要、数据集标识、源码版本、后端与版本、命令、状态
    weights/                         后端的检查点
    eval/                            测试划分上的指标与分片报告
    export/                          导出的 ONNX、labels.txt、签名与一致性检查结果
```

## 4. 准备数据

`train prepare <id>`：

1. 读取 `model.yaml` 的 `data.sources`，逐个来源读取图像与标注。
2. 类别名对照 `classes`：`ignore_classes` 中的丢弃，其他未知类别名报错（拼写错误不会悄悄变成新类别）；检查框是否越界、图像能否解码。
3. 按 `data.split` 划分：`seed` 固定，`group_by: filename_prefix` 让同一采集组（连续帧）整体进入同一划分，避免近似帧同时出现在训练与测试中；类别分布按组做分层平衡。
4. 写出 `manifest.json`，以它的摘要作为数据集标识。数据或划分参数不变时标识不变，`prepare` 直接复用已有数据集。

报告每个划分的样本数与各类别实例数；测试划分中缺少的类别给出警告（它的指标无法评估）。

## 5. 训练与评估

`train run <id> [--dataset <标识>]`（默认用最近一次 prepare 的结果）：

- 在训练镜像中调用 `train.backend` 指定的后端，`train.weights` 为起点，`train.options` 原样透传。
- run 标识为 `<UTC 时间>-<短摘要>`；`run.json` 记录 `model.yaml` 的摘要、数据集标识、源码版本（与打包相同的 `-dirty` 规则）、后端名称与版本，训练中断可续训。
- 训练结束后在测试划分上评估，写入 `eval/`：总体 mAP50、mAP50-95、每类 AP 与 `evaluate.slices` 定义的分片。

`train eval <id> <run>` 可单独重新评估，例如数据集更新后比较新旧 run。

## 6. 导出与发布

`train publish <id> <run>`，顺序不可调换：

1. **门槛**：`evaluate.gate.min` 中的每项指标达到下限；`regression` 把 `release.json` 中的当前模型在本次数据集的测试划分上重新评估，新模型的指定指标下降不超过 `max_drop`（没有当前模型时跳过）。
2. **导出**：按 `export` 导出 ONNX，按 `output_layout` 在输出端追加布局转换；生成 `labels.txt`。
3. **验证**：读取 ONNX 签名，对照 `input` 与 `output_layout`；用 onnxruntime 在测试样本上与后端原生推理对比检测结果（一致性阈值写进 `export/`），不一致即失败。
4. **写入**：替换 `models/<id>/artifacts/`，写 `release.json`（产物 sha256、签名、run、数据集标识、指标、发布时间），检查 nvinfer 配置的契约键；任何失败都不改动模型目录。
5. 提交由人完成：审阅 `release.json` 中的指标与差异后提交。

`release.json` 不保存权重本身，只记录权重的 sha256 与 run 标识：权重留在训练根目录，用于续训与复现。

## 7. 打包与安装

- `scripts/build-release.sh` 在渲染之后、检查镜像之前执行模型校验（也可单独运行 `framework/deploy/dsf-deploy.py check-models`）：规格完整；规格的类别、输入、输出布局与发布记录一致；产物与 sha256 一致且没有未发布的文件；`labels.txt` 与类别一致；导出签名符合输入尺寸与已知输出布局；nvinfer 配置引用的 ONNX 与标签是已发布产物，`model-engine-file` 在 `/data/engines/<id>/` 下，`num-detected-classes` 等于类别数，固定 batch 导出时 `batch-size` 与之相等。
- 模型随发布包进入安装目录的 `models/`；`model-engine-builder` 在目标 GPU 上构建 engine，Jetson 与服务器各自构建，engine 不进包。

## 8. 后续：只更新模型

模型比程序更新得频繁。规划中的模型包只包含 `models/<id>/`：

- 预检：完整性、模型校验、输出布局与已安装的解析插件匹配。
- **停服之前构建 engine**：engine 按指纹存放在 `/data/engines`，可以在旧服务继续运行时用新模型预构建；随后只重启引用该模型的服务，停机时间从 TensorRT 构建的数分钟缩短到重启的数秒。
- 保留上一份模型目录，回滚即切回并重启。

INT8：`prepare` 可从训练划分抽取校准图像，发布时生成校准缓存并作为产物发布（`int8-calib-file` 已计入 engine 指纹）。

数据闭环：现场应用已经发布检测结果并录像，可以增加一个服务按规则（低置信度、类别混淆、人工标记）抽取帧与预测结果，作为预标注导入新的 `sources/`，回到 §4。

## 9. 实现顺序

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 模型目录格式、`release.json`、打包前的模型校验 | 已实现 |
| 2 | 训练镜像与 `train prepare`（导入、校验、划分、数据集标识） | 待实现 |
| 3 | `train run` / `train eval`（Ultralytics 后端） | 待实现 |
| 4 | `train publish`（门槛、导出、一致性验证、写入模型目录） | 待实现 |
| 5 | 模型包与停服前构建 engine | 待实现 |
| 6 | INT8 校准、数据闭环 | 待实现 |

训练镜像使用 PyTorch 官方 CUDA 镜像加训练后端，只在 x86 GPU 训练机运行。Ultralytics 以 AGPL-3.0 发布：它只存在于训练镜像中，不进入 base、apps 或发布包；导出的 ONNX 是否受其许可约束，商用前需要单独确认。
