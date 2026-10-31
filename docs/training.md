# 模型训练与发布

本文说明从数据到现场运行的模型生命周期：导入图片或视频 → 标注 → 准备数据集 → 训练与评估 → 导出与发布 → 打包安装 → 在服务器或 Jetson 上运行。

已经实现：模型目录格式与打包前的模型校验（§3、§10），训练镜像与 `train init`、`train import`、`train accept`、`train prepare`（§5、§6），`train run`、`train eval`（§7）；均为一级检测。其余命令按 §14 的顺序实现；分类、分割与二级模型按 §11、§12 留出扩展点。

## 1. 原则

| 原则 | 含义 |
|---|---|
| 规格与事实分开 | `model.yaml` 是人写的规格；`release.json` 是工具写的发布事实，不手改 |
| 内容决定标识 | 数据集与发布产物用内容摘要标识；同样的输入得到同样的标识，任何改动都得到新标识 |
| 只有发布的东西进仓库 | 原始数据、标注、数据集、训练过程与权重留在训练根目录；仓库只有规格、发布记录与部署所需的产物 |
| 人只在两处做判断 | 标注（在 X-AnyLabeling 中画框、修正预标注）与发布（审阅指标后提交）；其余步骤可重复执行，结果确定 |
| 只收人确认过的标注 | 只有在 X-AnyLabeling 中标为已检查（`checked: true`）的图片进入数据集；预标注与半成品都不进入 |
| 测试集只增不换 | 新数据加入时，已有样本的划分不变：新旧模型总能在同一批没训练过的样本上比较 |
| 契约在打包前检查 | 类别、输入尺寸、输出布局、标签、级联关系与 nvinfer 配置不一致时不产出发布包，而不是到现场才发现 |
| 训练不进运行环境 | 训练镜像（PyTorch、训练框架）独立于 base/apps，不进入发布包 |
| 任务与后端可替换 | 检测、分类、分割是任务适配器，Ultralytics 等是后端适配器；规格中的 `options` 原样透传给后端 |

## 2. 典型流程

命令在训练机上通过 `scripts/train.sh <命令>` 运行（训练镜像中执行，见 §9）。

**从零开始训练第一版：**

```text
train init road-assets --task detect          生成 models/road-assets/model.yaml，填写类别
train import road-assets ~/raw/2026-11/ --source batch-01
                                              图片去重、视频抽帧，写入 sources/batch-01/
（在 X-AnyLabeling 中打开 sources/batch-01/ 手工标注，可用工具内置的通用模型辅助，标完一张勾选一张）
train prepare road-assets                     只收已检查的图片，生成数据集
train run road-assets                         训练 v0，测试集上评估
train prelabel road-assets --source batch-02 --run <v0 的 run>
                                              用尚未发布的 v0 预标注下一批，人只需修正并勾选
（修正 batch-02 → prepare → run，迭代到指标满意）
train publish road-assets <run>               门槛 → 导出 → 验证 → 写入 models/road-assets/
git commit && scripts/build-release.sh        提交、打包（模型校验），现场 ./install.sh
```

**已有发布模型，加一批新数据更新：**

```text
train import road-assets <新图片或视频> --source batch-07
train prelabel road-assets --source batch-07   默认用当前发布的模型
（在 X-AnyLabeling 中修正并勾选）
train prepare road-assets && train run road-assets
train publish road-assets <run>               与当前模型在同一测试集上比较，退化则拒绝
```

## 3. 模型目录（仓库内）

```text
models/<id>/
  model.yaml          规格：任务、类别、输入尺寸、输出布局、级联关系、数据、训练配方、发布门槛、导出参数
  release.json        发布事实：产物 sha256、导出的 ONNX 签名、来源 run 与数据集、指标（发布工具写入）
  artifacts/          已发布的产物：model.onnx、labels.txt（只能是 release.json 列出的文件）
  nvinfer.txt         nvinfer 配置：人工维护的运行参数，引用 artifacts/ 中的产物
  nvinfer.trtexec     可选：engine 构建的额外 trtexec 参数（见 deployment.md §4）
```

- `task` 是 `detect`、`classify` 或 `segment`（§11），决定标注怎样解读、数据集与指标是什么、nvinfer 配置应是哪种网络。
- `classes` 顺序即类别号。发布之后调换或删减类别会改变已部署业务与下游二级模型对类别号的解读，所以规格与发布记录不一致时校验失败：要么重新训练发布，要么恢复原值。新增类别追加在末尾，同样需要重新发布。
- `input` 与 `output_layout` 是部署契约。`output_layout` 由解析插件定义，例如 `plugins/yolo-parser` 接受 `xyxy-confidence-class_id`（唯一输出层 `[batch, 候选数, 6]`）。
- `cascade`（可选）声明这是二级模型及它作用于一级模型的哪些类别（§12）。
- 应用配置引用 `${CONTAINER_MODELS_PATH}/<id>/nvinfer.txt`；engine 由 `model-engine-builder` 按 `model-engine-file` 在目标机构建，产物更新后指纹变化，自动重建。`model-engine-file` 必须在 `/data/engines/<id>/` 下：模型目录对所有容器只读，数据目录跨发布与清洁安装保留。
- 一个模型目录可以有多份 nvinfer 配置（例如不同精度），每份都按同样规则检查。

产物目前直接提交到仓库。模型频繁重新发布时，ONNX 会让仓库历史持续变大，可以给 `models/*/artifacts/**` 启用 Git LFS（校验与打包只看文件内容，不受影响）。

## 4. 训练根目录（仓库外）

```text
<TRAIN_ROOT>/                         默认 ~/<项目名>/training，可用 TRAIN_ROOT 指定（建议放在大容量盘）
  sources/<来源>/                     一批导入的素材，X-AnyLabeling 直接打开这个目录
    *.jpg + *.json                    图像与同名的 X-AnyLabeling 标注（工具默认把标注存在图像旁边）
    source.json                       导入记录：每张图来自哪个文件/视频的哪个时刻、所属采集组、sha256
    labels.txt                        给 X-AnyLabeling 导入的类别清单（来自 model.yaml，避免手输类别名）
    prelabel.json                     预标注记录：哪些标注文件由哪个模型写入（追溯与统计人工修正量，不决定收录）
  datasets/<id>/<数据集标识>/          准备好的数据集，不可变
    manifest.json                     每个样本的图像 sha256、解读后的标注、所属划分；数据集标识 = 它的摘要
    view/<后端>/                       后端需要的视图（硬链接到 sources，不复制图像）
  runs/<id>/<run>/                    一次训练；完成后 train/ 不再改变
    run.json                          规格摘要、数据集标识、源码版本、后端与版本、起点权重、生效参数、命令、状态
    model.yaml                        训练时规格的副本
    train/                            后端的输出：检查点（weights/best.pt、last.pt）、曲线与日志
    eval/<数据集标识>/                 该数据集测试划分上的 metrics.json 与后端的图表
    export/                           导出的 ONNX、labels.txt、签名与一致性检查结果
  pretrained/                         下载的预训练权重（后端的其他下载物也在这里），下载一次，所有模型复用
```

一个来源只属于一次导入；来源可以被多个模型的 `data.sources` 引用（例如一级检测与基于它的二级分类共用同一批标注）。

## 5. 素材进入与标注

### 5.1 导入：`train import <id> <路径…> --source <来源>`

- **图片**：复制（同一文件系统时硬链接）到 `sources/<来源>/`；按 sha256 去掉完全相同的图片，与已有来源重复的图片报告并跳过，避免同一张图同时出现在训练与测试中。
- **视频**：按 `--every`（默认 1 秒）抽帧；用 64 位差值哈希去掉与上一张保留帧几乎相同的帧（`--dedupe`，默认相差不超过 4 位即视为相同，`-1` 保留全部帧），静止画面不会产生几百张一样的图；`--max-frames` 限制每个视频的帧数。帧命名为 `<视频名>_<毫秒>.jpg`。
- **采集组**：同一视频的帧属于同一组；图片默认以所在子目录为组（`--group-by filename_prefix` 改用文件名前缀，适合按序号命名的连拍）。组决定划分（§6），近似帧不会被拆到训练与测试两边。
- 写出 `source.json` 与 `labels.txt`。导入不改动原始文件，`source.json` 记录来源路径，便于追溯。

### 5.2 标注：X-AnyLabeling

在 X-AnyLabeling 中打开 `sources/<来源>/`，先导入 `labels.txt` 作为预设类别。标注保存为图像旁的同名 JSON（LabelMe 兼容格式），以 4.0.6 实际保存的文件为准（样例见 `framework/tests/python/data/x-anylabeling-4.0.6/`）：

- 顶层：`imagePath`、`imageWidth`、`imageHeight`、`shapes`、图像级 `flags`、`checked`（工具中的“已检查”状态）；`imageData` 为空（图像不嵌入 JSON，非空时忽略）。
- 每个形状：`label`、`points`、`shape_type`、`difficult`、`attributes`（形状属性，例如 `{"遮挡": "否", "破损": "否"}`），以及读取时忽略的 `group_id`、`description`、`flags`、`kie_linking`。

各任务读取的内容：

| 任务 | 读取 | 说明 |
|---|---|---|
| 检测 | `shape_type: rectangle` | 取 `points` 的外接矩形（4.0.6 写四个角点，也兼容两点写法）；`rotation`（旋转框）留给 OBB 任务 |
| 分割 | `shape_type: polygon` | 实例多边形；SAM 辅助画出的轮廓也是多边形 |
| 分类（整图） | `flags` 中被勾选的类别 | 每张图恰好一个类别 |
| 二级分类 | 一级类别框的 `attributes` 中的指定属性 | 一次标注同时得到一级框与二级标签（§12） |

- **只收 `checked: true` 的图片。** 工具里的“已检查”是人对这张图的唯一确认：标完一张勾选一张。没有勾选的图片（还没标完、被打断、预标注还没修正）一律不进入，`prepare` 按来源报告未检查的数量，忘了勾选的图片不会悄悄缺席。
- 已检查且没有形状的图片是**负样本**（确认画面中没有目标），进入数据集；未检查的空标注可能只是漏标，不能当作背景。没有 JSON 的图片是**未标注**。
- 已检查的图片后来被修改并保存时，`checked: true` 保留（4.0.6 已核实）：视为修改的人确认了修改后的结果，照常收入。所以改动已检查的图片要像新标注一样认真，改到一半就保存的结果也会进入数据集。
- X-AnyLabeling 4.0.6 只能逐张勾选。需要批量确认时（主要是勾选约定之前的旧标注）用 `train accept <来源> [文件…]`：它把指定文件的 `checked` 改为 `true`，并在 `source.json` 中记录哪些文件是批量确认的及时间；这是显式的一步，确认的责任在执行它的人。
- 标签不在 `classes` 中、也不在 `data.ignore_classes` 中时 `prepare` 报错，并列出文件名，拼写错误不会悄悄变成新类别。
- 任务不使用的形状（例如检测任务中的多边形）报告为警告并忽略，不报错：同一来源可能同时服务于检测与分割。
- 形状属性随样本记入 manifest：它是二级模型的标签来源（§12），也可以定义评估分片（例如只看“遮挡=是”的目标，§7）。
- `difficult` 记入 manifest；训练照常使用，评估时可以按它分片。

读取器的测试使用真实保存的样例文件；升级 X-AnyLabeling 版本时补充该版本的样例。

### 5.3 第一版模型：还没有任何发布

没有可用于预标注的自有模型时：

1. 先手工标注一小批，覆盖各类别与主要场景（经验上每类至少几十个实例）。X-AnyLabeling 内置的通用模型可以在工具里直接辅助：按类别名做文字提示的开放词汇检测（如 YOLO-World、Grounding DINO），用 SAM 点选生成框或多边形。它们在工具内运行，不经过本框架。
2. `train run` 不要求模型已经发布：训练出 v0 后，用 `train prelabel --run <run>` 预标注下一批，修正、`prepare`、再训练。这样每一轮人只修正而不重画，直到指标满意才发布。
3. 首次 `publish` 没有可对照的旧模型：只检查 `evaluate.gate.min`。`min` 为空时 `publish` 要求显式的 `--accept-metrics`，避免第一版在没有任何门槛的情况下发布。

`train init <id> --task <任务>` 生成该任务的 `model.yaml` 骨架（含后端默认的训练配方），填好类别后即可导入素材。

### 5.4 预标注：`train prelabel <id> --source <来源> [--run <run>]`

- 默认用 `release.json` 中的当前模型，`--run` 改用某次训练（包括未发布的）；在训练镜像中用后端原生推理，置信度阈值 `--conf`。
- 只给**还没有 JSON** 的图片写标注，永不覆盖已有标注；写入的 JSON 与 X-AnyLabeling 保存的格式相同，打开即可修正。
- 写入的标注一律 `checked: false`：人在工具中修正（预测正确时直接）勾选后才进入数据集，与人工标注同一条规则（§5.2），不需要另外的确认机制。
- 写入的文件与所用模型（发布或 run）记入 `prelabel.json`，用于追溯，并统计人工修正了多少预测框（衡量模型在新数据上的表现）；它不影响收录。

## 6. 准备数据集：`train prepare <id>`

1. 读取 `model.yaml` 的 `data.sources`，按任务解读每个来源中已检查的标注（§5.2），检查框是否越界、多边形是否有效、图像能否解码。
2. **划分按采集组的哈希决定**：组标识与 `data.split.seed` 的摘要落在哪个区间，就进入训练、验证或测试，与其他样本无关。加入新来源时，已有的组划分不变；测试集只会增加新的组，旧模型从未训练过的测试样本一直留在测试集中，`publish` 的退化比较才有意义。代价是各类别在划分间的比例只能近似，`prepare` 报告每个划分的各类别实例数，测试集中缺少的类别给出警告（它的指标无法评估）。
3. 写出 `manifest.json` 与后端视图，以 manifest 的摘要作为数据集标识。素材、标注与划分参数都不变时标识不变，直接复用已有数据集。
4. 报告：收入与跳过的样本（未标注、未检查、未知类别），各划分的样本、组与实例数。

## 7. 训练与评估

`train run <id> [--dataset <标识>] [--option 键=值 …]`（默认用最近一次 prepare 的结果）：

- 在训练镜像中调用 `train.backend` 指定的后端，`train.options` 原样透传；没有给出 `imgsz` 时按 `input` 的尺寸训练。数据集、输出位置与续训由工具决定，写在 `options` 中（`data`、`project`、`name`、`resume` 等）是规格错误。
- `train.weights` 是起点：预训练权重的文件名（在 `pretrained/` 中查找，没有时由后端下载），`run:<run>`（某次完成的 run 的最佳权重，在已有模型上用新数据继续训练），或绝对路径。起点的 sha256 记入 `run.json`。
- `--option` 只覆盖这一次 run 的参数，例如试跑 `--option epochs=3`、选卡 `--option device=1`（多卡 `device=0,1`）；覆盖值单独记入 `run.json`，长期有效的参数写进 `model.yaml`。
- run 标识为 `<UTC 时间>-<短摘要>`（摘要来自规格、数据集与覆盖参数）；`run.json` 记录 `model.yaml` 的摘要与副本、数据集标识、类别、源码版本（与打包相同的 `-dirty` 规则）、后端与依赖版本、生效参数、命令与状态（`running`、`completed`、`failed`、`interrupted`）。
- 数据集必须按当前规格的类别准备：类别改动后先重新 `prepare`。验证划分为空时拒绝训练。
- 中断（Ctrl-C）或失败的 run 保留记录与最近检查点，`train run <id> --resume <run>` 沿用原来的数据集与参数继续。
- 训练结束后在测试划分上评估（测试划分为空时只给出警告），写入 `eval/<数据集标识>/metrics.json`：指标由任务决定（检测：`map50_95`、`map50`、`precision`、`recall` 及每类的同名指标与测试实例数；分类：top-1、每类精确率与召回率；分割：框与掩码的 mAP）。测试划分中没有实例的类别不给指标，单独列出。`evaluate.gate` 只能引用任务定义的指标名，写错时在训练前报告。

`train eval <id> <run> [--dataset <标识>]` 单独重新评估，例如数据集更新后在新的测试划分上比较新旧 run；每个数据集的结果分别保存，摘要记入 `run.json` 的 `evaluations`。

分片评估（`evaluate.slices`：按来源、按形状属性，例如“遮挡=是”的目标，或按 `difficult`）尚未实现：按来源是图片级的子集，按属性是目标级的子集，后者需要在评估时忽略其余目标，等有实际需要时再设计。

## 8. 导出与发布：`train publish <id> <run>`

顺序不可调换：

1. **门槛**：`evaluate.gate.min` 中的每项指标达到下限；`regression` 把 `release.json` 中的当前模型在本次数据集的测试划分上重新评估，新模型的指定指标下降不超过 `max_drop`（没有当前模型时跳过，见 §5.3）。
2. **导出**：按 `export` 导出 ONNX，按任务与 `output_layout` 在输出端追加布局转换；生成任务要求格式的 `labels.txt`。
3. **验证**：读取 ONNX 签名，对照 `input` 与 `output_layout`；用 onnxruntime 在测试样本上与后端原生推理对比结果（一致性阈值写进 `export/`），不一致即失败。
4. **写入**：替换 `models/<id>/artifacts/`，写 `release.json`（产物 sha256、签名、run、数据集标识、指标、发布时间）；把 `model-engine-file` 中的发布段改为本次的发布标识（`/data/engines/<id>/<发布>/…`），新旧 engine 并存，回滚不需要重建；检查 nvinfer 配置的契约键。任何失败都不改动模型目录。
5. 提交由人完成：审阅 `release.json` 中的指标与差异后提交。

`release.json` 不保存权重本身，只记录权重的 sha256 与 run 标识：权重留在训练根目录，用于续训、预标注与复现。

## 9. 训练机与镜像

- 训练镜像（`framework/docker/train.Dockerfile`）基于 PyTorch 官方 CUDA 镜像加训练后端与 onnx/onnxruntime，版本固定，只在 x86 GPU 训练机运行。镜像只提供运行环境，训练工具（`framework/training/`）不复制进镜像：依赖（含间接依赖）全部固定在 `framework/docker/train-requirements.txt`；`scripts/train.sh` 以 Dockerfile 与依赖文件的摘要作为 tag，缺少时自动构建；以当前用户运行，挂载仓库（可写）、训练根目录（可写）与当前目录（只读），参数中含 `/` 的已有路径按真实路径只读挂载。
- Ultralytics 以 AGPL-3.0 发布：它只存在于训练镜像中，不进入 base、apps 或发布包；导出的 ONNX 是否受其许可约束，商用前需要单独确认。
- 不依赖 GPU 的部分（标注读取、去重、划分、manifest、规格与发布记录）是纯 Python，在 builder 的 CTest 中测试；训练、导出与一致性验证只在训练机上验证。

## 10. 打包与安装

- `scripts/build-release.sh` 在渲染之后、检查镜像之前执行模型校验（也可单独运行 `framework/deploy/dsf-deploy.py check-models`）。已实现的检查：规格完整；规格的类别、输入、输出布局与发布记录一致；产物与 sha256 一致且没有未发布的文件；`labels.txt` 与类别一致；导出签名符合输入尺寸与已知输出布局；nvinfer 配置引用的 ONNX 与标签是已发布产物，`model-engine-file` 在 `/data/engines/<id>/` 下，`num-detected-classes` 等于类别数，固定 batch 导出时 `batch-size` 与之相等。
- 随任务与级联加入的检查（§11、§12）：`network-type` 与任务一致；分类模型的 `labels.txt` 格式；二级模型的 `operate-on-gie-id` 与 `operate-on-class-ids` 与一级模型对得上。
- 模型随发布包进入安装目录的 `models/`；`model-engine-builder` 在目标 GPU 上构建 engine，Jetson 与服务器各自构建，engine 不进包。

## 11. 任务扩展

任务适配器决定一个任务从标注到部署的每一步；新增任务只加适配器与对应的解析插件，不改流程：

| 任务 | 标注 | 后端视图（Ultralytics） | 指标 | nvinfer | 输出与解析 | 状态 |
|---|---|---|---|---|---|---|
| detect | 矩形框 | YOLO 检测格式 | mAP | `network-type=0` | `xyxy-confidence-class_id`，`plugins/yolo-parser` | 首先实现 |
| classify | 整图 flags 或二级属性 | 按类别分目录 | top-1、每类 P/R | `network-type=1` | softmax，nvinfer 内置分类解析；`labels.txt` 为一行、以 `;` 分隔 | 预留 |
| segment | 多边形 | YOLO 分割格式 | 框与掩码 mAP | `network-type=3` | 实例掩码，需要配套解析插件 | 预留 |
| obb、pose 等 | 旋转框、关键点 | 后端对应格式 | 后端对应指标 | 视解析插件 | 视解析插件 | 不在计划内 |

适配器接口（`framework/training` 中，任务与后端各一组）：

- 任务：从一份标注解读出样本（并报告不认识的形状与类别）；写出后端视图；指标的名称与比较方向；导出时追加的输出转换；`labels.txt` 的格式；发布时与打包时的契约检查。
- 后端：训练、评估、原生推理（预标注与一致性验证用）、导出 ONNX；声明支持的任务。

还没有实现的任务在 `train init` 与 `model.yaml` 校验时明确报告“任务尚未支持”，而不是走到训练才失败。

## 12. 一级与二级模型

二级模型作用于一级模型检测到的目标（例如在“摄像头”框内再分类状态），在 `model.yaml` 中声明：

```yaml
task: classify
cascade:
  parent: road-assets                   # 一级模型的 id
  classes: [摄像头(柱), 摄像头(球机)]     # 父模型中触发本模型的类别，按名称写
data:
  from_parent:
    attribute: 状态                      # 训练样本 = 父类别框的裁剪图，类别 = 框上该属性的值
    padding: 0.1                         # 裁剪时向外扩展的比例
    min_size: 32                         # 小于它的框不收入（现场同样太小，无法可靠分类）
```

- **数据**：二级模型复用父模型的来源。X-AnyLabeling 的形状属性让一次标注同时得到一级框与二级标签；`prepare` 从父类别框裁剪出样本，划分沿用这些框所在图片的采集组，两级模型的测试集来自同一批图片。也可以单独标注裁剪图作为普通分类来源。
- **评估**：先在真实框的裁剪上评估二级模型本身；端到端评估（一级检测框 → 二级分类）以后按需要增加。
- **部署契约**：二级模型的 nvinfer 配置使用 `process-mode=2`，`operate-on-gie-id` 等于父模型配置的 `gie-unique-id`，`operate-on-class-ids` 由 `cascade.classes` 按父模型 `classes` 换算出类别号。打包时校验这两者：父模型重新发布并改变了类别号时，二级模型的配置在打包前失败，而不是在现场作用于错误的类别。
- 应用配置的 `inference` 链按顺序串接一级与二级模型（已支持）。

## 13. 后续：只更新模型

模型比程序更新得频繁。规划中的模型包只包含 `models/<id>/`：

- 预检：完整性、模型校验、输出布局与已安装的解析插件匹配。
- **停服之前构建 engine**：engine 路径带发布标识（§8），可以在旧服务继续运行时用新模型预构建；随后只重启引用该模型的服务，停机时间从 TensorRT 构建的数分钟缩短到重启的数秒。不再被任何配置引用的发布目录由 engine 构建工具清理。
- 保留上一份模型目录，回滚即切回并重启。

INT8：`prepare` 可从训练划分抽取校准图像，发布时生成校准缓存并作为产物发布（`int8-calib-file` 已计入 engine 指纹）。

数据闭环：现场应用已经发布检测结果并录像，可以增加一个服务按规则（低置信度、类别混淆、人工标记）抽取帧与预测结果，作为预标注导入新的来源，回到 §5.4。

## 14. 实现顺序

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 模型目录格式、`release.json`、打包前的模型校验 | 已实现 |
| 2 | 训练镜像、`scripts/train.sh`；`train init`、`train import`（图片去重、视频抽帧）；X-AnyLabeling 读取与 `train accept`；`train prepare`（检测） | 已实现 |
| 3 | `train run` / `train eval`（Ultralytics，检测）；分片评估除外 | 已实现 |
| 4 | `train prelabel` | 待实现 |
| 5 | `train publish`（检测：门槛、导出、一致性验证、按发布命名 engine） | 待实现 |
| 6 | 模型包与停服前构建 engine | 待实现 |
| 7 | 分类与分割任务、二级模型及其打包校验 | 预留 |
| 8 | INT8 校准、数据闭环 | 预留 |

待设计（阶段 3 之后再定）：修改已有模型的类别（增加、删除、改名）。目前只能手改 `model.yaml`，已知的问题：

- 新增类别后，之前已检查的图片里没有新类别的框，`prepare` 仍会收入，训练与测试都把新类别当作背景。X-AnyLabeling 只有 `checked` 一个确认信号，看过却没改的图片不会留下痕迹，所以要求复核大概只能撤销 `checked`；复核量大时人会倾向 `accept --all`，需要考虑如何防止复核流于形式。曾讨论的方案：`train update` 先在 `source.json` 记录变更与被撤销的文件，再撤销 `checked`、追加类别、刷新 `labels.txt`；共享来源时提示受影响的其他模型；可显式跳过确定不含新类别的来源。
- 已有来源的 `labels.txt` 只在 `import` 时写入，类别变化后不会更新。
- 已发布模型改变类别后，打包校验失败，直到重新发布；删除与改名会改变类别号，影响已部署业务与二级模型。
- 预标注（阶段 4）可以只为新类别在已有标注上补框，减少复核量。
