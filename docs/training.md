# 模型训练与发布

本文说明从数据到现场运行的模型生命周期：导入图片或视频 → 标注 → 准备数据集 → 训练与评估 → 导出与发布 → 打包安装 → 在服务器或 Jetson 上运行。

已经实现：模型目录格式与打包前的模型校验（§3、§10），训练镜像与 `train init`、`train import`、`train accept`、`train prepare`（§5、§6），`train run`、`train eval`（§7），`train prelabel`（§5.4），`train publish`（§8），一级检测与二级分类（§12）均可。其余按 §14 的顺序实现；整图分类与分割按 §11 留出扩展点。

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
train publish road-assets <run>                与当前模型在同一测试集上比较，退化则拒绝
```

## 3. 模型目录（仓库内）

```text
models/<id>/
  model.yaml          规格：任务、类别、输入尺寸、输出布局、级联关系、数据、训练配方、发布门槛、导出参数
  release.json        发布事实：产物 sha256、导出的 ONNX 签名、来源 run 与数据集、指标（发布工具写入）
  artifacts/          已发布的产物：model.onnx、labels.txt（只能是 release.json 列出的文件）
  nvinfer.txt         nvinfer 配置：人工维护的运行参数，引用 artifacts/ 中的产物
  nvinfer.trtexec     可选：engine 构建的额外 trtexec 参数（见 deployment.md §4）
  sources.lock.json   确认过的各来源类别列表（工具写入，§15.1）
```

- `task` 是 `detect`、`classify` 或 `segment`（§11），决定标注怎样解读、数据集与指标是什么、nvinfer 配置应是哪种网络。
- `classes` 顺序即类别号。发布之后调换或删减类别会改变已部署业务与下游二级模型对类别号的解读，所以规格与发布记录不一致时校验失败：要么重新训练发布，要么恢复原值。修改类别用 `train update`，见 §15。
- `input` 与 `output_layout` 是部署契约。`output_layout` 由解析插件定义，例如 `plugins/yolo-parser` 接受 `xyxy-confidence-class_id`（唯一输出层 `[batch, 候选数, 6]`）。
- `cascade`（可选）声明这是二级模型及它作用于一级模型的哪些类别（§12）。
- 应用配置引用 `${CONTAINER_MODELS_PATH}/<id>/nvinfer.txt`；engine 由 `model-engine-builder` 按 `model-engine-file` 在目标机构建，产物更新后指纹变化，自动重建。`model-engine-file` 必须在 `/data/engines/<id>/` 下：模型目录对所有容器只读，数据目录跨发布与清洁安装保留。
- 一个模型目录可以有多份 nvinfer 配置（例如不同精度），每份都按同样规则检查。

ONNX 用 Git LFS 存放（`.gitattributes` 中的 `models/*/artifacts/*.onnx`）：模型频繁重新发布时，仓库历史不随每个版本增大。`labels.txt` 等文本产物照常进入 git，类别变化在评审中直接可见。克隆、训练、打包的机器都需要 git-lfs（`git lfs install`）；没有它时检出的 ONNX 是指针文件，打包前的模型校验报告产物与 sha256 不符而拒绝打包，不会把指针发出去。启用之前提交的 ONNX 仍在普通历史中，没有改写。

## 4. 训练根目录（仓库外）

```text
<TRAIN_ROOT>/                         默认 ~/<项目名>/training，可用 TRAIN_ROOT 指定（建议放在大容量盘）
  sources/<来源>/                     一批导入的素材，X-AnyLabeling 直接打开这个目录
    *.jpg + *.json                    图像与同名的 X-AnyLabeling 标注（工具默认把标注存在图像旁边）
    source.json                       导入记录：每张图来自哪个文件/视频的哪个时刻、所属采集组、sha256；类别与复核名单（§15）
    labels.txt                        给 X-AnyLabeling 导入的类别清单（来源的类别，避免手输类别名）
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
- **来源的类别**：新来源取 `<id>` 的类别，记入 `source.json` 的 `classes`，`labels.txt` 是它的副本。向已有来源继续导入时来源的类别不变（其他模型可能也在用它），`<id>` 的类别必须都在其中，否则拒绝导入；给来源增加类别用 `train update`（§15）。

### 5.2 标注：X-AnyLabeling

在 X-AnyLabeling 中打开 `sources/<来源>/`，先导入 `labels.txt` 作为预设类别。要标形状属性（二级模型的标签，§12）时载入 `sources/<来源>/attributes.json`：Image Classifier（Ctrl+3）的 MultiTask 中对象选择框旁的上传按钮。它和 `labels.txt` 一样由工具生成、不手写，也没有单独的命令：`train import`、`train prepare`（在读取标注之前，所以 prepare 因标注问题失败时它已是最新）与改写来源的 `train update` 顺带重新生成，输出中给出它的位置。内容：二级模型声明的属性（`data.from_parent.attribute`）出现在它的 `cascade.classes` 上，取值为 `classes` 加 `ignore_classes`；来源标注中已有的属性全部保留（文件中缺少的属性在工具中看不到）；同一来源中同名属性的取值取各类别上的并集，模型没有声明的取值按字符排序，重新生成时顺序不变。标注中有模型不接受的取值时给出警告（`prepare` 会拒绝它们）。暂时不成立的二级模型规格（例如父类别改名后 `cascade.classes` 还没改写）不使命令失败，只是这次不计入它的属性，并说明原因。修改二级模型的属性或取值后运行一次 `train prepare <模型>` 即可刷新。标注保存为图像旁的同名 JSON（LabelMe 兼容格式），以 4.0.6 实际保存的文件为准（样例见 `framework/tests/python/data/x-anylabeling-4.0.6/`）：

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
- X-AnyLabeling 4.0.6 只能逐张勾选。需要批量确认时（主要是勾选约定之前的旧标注）用 `train accept <来源> [文件…]`：它把指定文件的 `checked` 改为 `true`，并在 `source.json` 中记录哪些文件是批量确认的及时间；这是显式的一步，确认的责任在执行它的人。新增类别后等待复核的图片（§15.2）要另给 `--reason`。
- 来源中有、模型不用的类别（其他模型的类别）在读取时忽略。标签既不在模型的类别中、也不在来源的类别与 `data.ignore_classes` 中时 `prepare` 报错，并列出文件名，拼写错误不会悄悄变成新类别。
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

- 默认用 `release.json` 中的当前模型：按记录的 run 在训练根目录中找到权重，并核对 sha256（权重不在这台训练机上时报错，改用 `--run`）。`--run` 改用某次完成的训练（包括未发布的）。所用模型的类别都要在当前规格中（按名称），写出它认识的类别：新增类别之后，当前发布的模型仍可预标注旧类别。在训练镜像中用后端原生推理，置信度阈值 `--conf`（默认 0.25），输入尺寸沿用该 run 的参数，设备按 `--device`、`TRAIN_DEVICE`、run 的设备依次取第一个（§7）。NMS 不区分类别：一个目标只写一个框，相邻候选判成不同类别时只保留得分最高的那个。
- 只给**还没有 JSON** 的图片写标注，永不覆盖已有标注：推理期间人在工具中保存了某张图时，跳过这张。写入的 JSON 与 X-AnyLabeling 4.0.6 保存的字段与键顺序相同，打开即可修正；预测为空的图片也写出空标注，表示模型认为画面中没有目标。
- 写入的标注一律 `checked: false`：人在工具中修正（预测正确时直接）勾选后才进入数据集，与人工标注同一条规则（§5.2），不需要另外的确认机制。
- 写入的文件与所用模型（发布或 run、权重 sha256、阈值）记入 `prelabel.json`，用于追溯；它不影响收录。中断后再次运行只处理剩下没有 JSON 的图片。
- `train prepare` 对引用的来源报告每批预标注的修正量：已检查的图片中，预测框保留（同类别且 IoU ≥ 0.5，允许微调位置）、删除（或改了类别）与人补充的框各有多少，衡量模型在新数据上的表现。

## 6. 准备数据集：`train prepare <id>`

1. 检查类别依赖（§15.1）：模型的类别必须都在每个来源的类别中；来源的类别必须与 `sources.lock.json` 的记录相同，不同时打印差异并拒绝，记录中没有的来源按当前类别补记并提示。`run` 与 `publish` 做同样的检查（不补记）。
2. 读取 `model.yaml` 的 `data.sources`，按任务解读每个来源中已检查的标注（§5.2），检查框是否越界、多边形是否有效、图像能否解码。
3. **划分按采集组的哈希决定**：组标识与 `data.split.seed` 的摘要落在哪个区间，就进入训练、验证或测试，与其他样本无关。加入新来源时，已有的组划分不变；测试集只会增加新的组，旧模型从未训练过的测试样本一直留在测试集中，`publish` 的退化比较才有意义。代价是各类别在划分间的比例只能近似，`prepare` 报告每个划分的各类别实例数，测试集中缺少的类别给出警告（它的指标无法评估）。
4. 写出 `manifest.json` 与后端视图，以 manifest 的摘要作为数据集标识。素材、标注与划分参数都不变时标识不变，直接复用已有数据集。
5. 报告：收入与跳过的样本（未标注、未检查、未知类别），各划分的样本、组与实例数。

## 7. 训练与评估

`train run <id> [--dataset <标识>] [--option 键=值 …]`（默认用最近一次 prepare 的结果）：

- 在训练镜像中调用 `train.backend` 指定的后端，`train.options` 原样透传；没有给出 `imgsz` 时按 `input` 的尺寸训练。数据集、输出位置与续训由工具决定，写在 `options` 中（`data`、`project`、`name`、`resume` 等）是规格错误。
- `train.weights` 是起点：预训练权重的文件名（在 `pretrained/` 中查找，没有时由后端下载），`run:<run>`（某次完成的 run 的最佳权重，在已有模型上用新数据继续训练），或绝对路径。起点的 sha256 记入 `run.json`。
- `--option` 只覆盖这一次 run 的参数，例如试跑 `--option epochs=3`、选卡 `--option device=1`（多卡 `device=0,1`）；覆盖值单独记入 `run.json`，长期有效的参数写进 `model.yaml`。
- 类别不均衡（分类）：`train.class_weights` 给交叉熵损失按类别加权，少数类的错分代价更高。取 `balanced` 时按训练划分的样本数取反比（总数 / (类别数 × 该类样本数)），或写成 `{类别: 权重}`（没写的类别为 1）。生效的权重按类别号记入 `run.json` 的 `class_weights`，训练开始时打印。加权提高少数类的 recall、通常降低它的 precision；最佳检查点按宏平均 recall 选择。检测模型不接受这一项。不做过采样：训练时的随机增强下，加权与重复样本的效果相当，而加权不改变数据集与视图。
- 设备属于训练机而不是模型或 run：`model.yaml` 不写 `device`，在训练机上设置环境变量 `TRAIN_DEVICE`（例如 `export TRAIN_DEVICE=1`，多卡 `0,1`，由 `scripts/train.sh` 传入容器）。训练时 `--option device=…`、`model.yaml`、`TRAIN_DEVICE` 依次取第一个，生效的值记入 `run.json`；`--resume` 沿用 run 原来的设备。评估、`--errors`、预标注与发布门槛的评估按 `--device`、`TRAIN_DEVICE`、run 训练时的设备依次取第一个（多卡取第一张）：训练时用的卡之后可能被别的任务占用，不必每条命令都写 `--device`。
- run 标识为 `<UTC 时间>-<短摘要>`（摘要来自规格、数据集与覆盖参数）；`run.json` 记录 `model.yaml` 的摘要与副本、数据集标识、类别、源码版本（与打包相同的 `-dirty` 规则）、后端与依赖版本、生效参数、命令与状态（`running`、`completed`、`failed`、`interrupted`）。
- 数据集必须按当前规格的类别准备：类别改动后先重新 `prepare`。验证划分为空时拒绝训练。
- 中断（Ctrl-C）或失败的 run 保留记录与最近检查点，`train run <id> --resume <run>` 沿用原来的数据集与参数继续。
- 训练结束后在测试划分上评估（测试划分为空时只给出警告），写入 `eval/<数据集标识>/metrics.json`：指标由任务决定（检测：`map50_95`、`map50`、`precision`、`recall` 及每类的同名指标与测试实例数；分类：top-1、每类精确率与召回率；分割：框与掩码的 mAP）。测试划分中没有实例的类别不给指标，单独列出。`evaluate.gate` 只能引用任务定义的指标名，写错时在训练前报告。

`train eval <id> <run> [--dataset <标识>] [--device <设备>]` 单独重新评估，例如数据集更新后在新的测试划分上比较新旧 run；每个数据集的结果分别保存，摘要记入 `run.json` 的 `evaluations`。

`train eval <id> <run> --errors [--split test|val|train …] [--device <设备>]`（二级分类；`--split` 可重复）列出这些划分中分错的裁剪：标注的值、预测的值与置信度、来源图片与框，置信度高的错误排在前面——它们最可能是标错了。完整列表写入 `eval/<数据集标识>/errors-<划分>.json`，同时建立复核集 `reviews/<id>-<run>-<划分…>/`：只含这些图片（硬链接）与它们标注的副本，、列出要核对的框的 `checklist.txt`，以及来源的 `attributes.json`（图片都来自一个来源时）。在 X-AnyLabeling 中打开这个目录、按 D 逐张核对并保存，然后 `train review apply <名称>` 把改过的副本写回来源（`imagePath` 恢复为来源中的文件名）。标注只能复制、不能链接：X-AnyLabeling 4.0.6 保存时写临时文件再替换，链接会被换成复核目录中的新文件。来源标注自复制以来变了的不写回、报告冲突；有未应用修改的复核集不会被同名的新复核集替换；`train prepare` 提醒涉及它的来源中有未应用的复核；`train review` 列出全部复核集。改正时按图片判断，不要因为模型不同意就改测试划分的标注，否则测试指标虚高；训练划分上的结果偏乐观（模型见过这些样本），仍能找出与模型强烈不一致的标注。

分片评估（`evaluate.slices`：按来源、按形状属性，例如“遮挡=是”的目标，或按 `difficult`）尚未实现：按来源是图片级的子集，按属性是目标级的子集，后者需要在评估时忽略其余目标，等有实际需要时再设计。

## 8. 导出与发布：`train publish <id> <run> [--dataset <标识>] [--accept-metrics] [--device <设备>]`

不给 `<run>` 时只列出可以发布的 run，不发布任何东西：完成的 run 按完成时间从新到旧，带各自在数据集（默认最近一次 prepare）测试划分上的指标与训练条件中和现在不同的部分（`--option` 覆盖参数；run 保存的 `model.yaml` 副本中 `input`、`train` 段与现在不同的键，给出 run 当时的取值；训练用的数据集不是所列数据集时给出它的标识），并标出当前发布所用的 run；还没有在该数据集上评估的 run 指标为 `-`（`train eval` 补上），中断或失败的 run 单独列出。不默认发布最近的 run：最近的往往是 `--option epochs=3` 之类的试跑，而首次发布时门槛可能只有 `--accept-metrics`，选哪个 run 是人的判断。

顺序不可调换：

1. **门槛**（在数据集的测试划分上，默认最近一次 prepare）：`evaluate.gate.min` 中的每项指标达到下限（按任务定义的比较方向）；`regression` 把 `release.json` 中的当前模型在同一测试划分上评估，新模型的指定指标下降不超过 `max_drop`。run 在该数据集上已有评估结果时直接使用（权重与数据集都不可变），`--device` 只用于需要新做的评估。
   - 没有当前模型（首次发布，§5.3）时跳过退化比较；此时 `min` 为空就没有任何门槛，要求显式的 `--accept-metrics`。
   - **测试实例少时按一个实例的影响放宽。** 一个测试实例能使指标变化多少由任务给出：各类平均的指标（检测的全部指标、二级分类的 `recall` 与 `precision`）是 1/(有测试实例的类别数 × 最稀有类别的实例数)，二级分类的 `accuracy` 是 1/裁剪总数。小于它的下降不是退化的证据（只是这一个实例），门槛允许 max(`max_drop`, 它)，在输出与 `release.json` 的 `resolution`、`allowed` 中写明是哪个类别、多少实例。例如测试划分中“遮挡=是”只有 22 个时，一个裁剪使宏平均 recall 变化 0.0227，0.01 的 `max_drop` 无从判断；要更严的门槛，就给测试划分增加这个类别的实例。
   - 当前模型无法重新评估（它的 run 不在这台训练机上、权重的 sha256 不符、类别已改变）时同样要求 `--accept-metrics`，而不是悄悄跳过比较。
   - `--accept-metrics` 只放行“没有可对照的东西”，未达到下限或退化超限时总是拒绝。门槛的结果（含是否靠 `--accept-metrics` 放行）写进 `release.json`。
2. **导出**：后端按 `export`（`format: onnx`、`opset`、`batch`、`dynamic`、`simplify`，`model.yaml` 加载时校验）以 `input` 的尺寸在 CPU 上导出 ONNX，并在输出端追加到 `output_layout` 的转换；输入与输出名固定为 `input`、`output`；去掉导出时间与含本机路径的描述，同一份权重总是导出同一个文件。导出的 ONNX 与当前发布相同时拒绝发布：新的发布标识只会让目标机为同一个模型重建 engine。生成任务要求格式的 `labels.txt`。导出物在 `runs/<id>/<run>/export/`，每次发布重新生成。
   - Ultralytics 检测头的原始输出是 `[batch, 4 + 类别数, 候选数]`（中心点、宽高与各类别分数）；追加的节点转置为每行一个候选，换算为左上右下，取最高分数与其类别号，得到 `xyxy-confidence-class_id` 的 `[batch, 候选数, 6]`。原生推理用同一组运算得到同一布局，两边可以逐行比较。
3. **验证**：读取 ONNX 签名，按打包时同一套规则对照 `input` 与 `output_layout`；在 CPU 上用 onnxruntime 与后端原生推理（PyTorch）对同一批测试样本（按 sha256 取前 16 张，letterbox 到输入尺寸）的输出逐行比较：置信度不低于 0.05 的候选，框坐标差不超过 0.5 像素、置信度差不超过 0.001，置信度不低于 0.25 的候选类别号相同，输出全部有限。阈值与结果写进 `export/export.json` 与 `release.json`，不一致即失败。比较的是导出本身；现场的 TensorRT engine（FP16）与预处理只能在目标机验证。
4. **写入**：发布标识为 `<UTC 时间>-<ONNX 摘要>`。先在暂存目录组装完整的新模型目录：`artifacts/` 只含本次的 `model.onnx` 与 `labels.txt`；`release.json` 记录发布标识、run（数据集、源码版本、后端版本、权重 sha256）、训练所用数据集的按来源统计（见下）、评估所用数据集、总体与每类指标、门槛结果、ONNX 签名、一致性结果与产物 sha256；各 nvinfer 配置的 `model-engine-file` 改为 `/data/engines/<id>/<发布>/<文件>`（还没有发布段时插入一段，已有的替换当前发布那一段，看不出哪一段是发布标识时报错）。组装结果通过打包时的模型校验（§10）后，才把变化的文件逐个原子替换进 `models/<id>/`，`release.json` 最后写入；校验失败时模型目录不变。新旧 engine 路径不同，目标机上新旧 engine 并存，回滚不需要重建。
5. 提交由人完成：审阅 `release.json` 中的指标与差异后提交。

按来源统计（`training_data`）：每个来源收入的图片数与其中的负样本数、每类实例数从 run 所用数据集的 manifest 计算，结果确定；按原因分开的跳过数（未标注、未检查、待复核 <类别>，§15.2）与尚未完成的复核名单（类别、时间、已重新勾选数/总数，文件列表在 `source.json`）是发布时来源的状态，说明这次发布时还有哪些素材没有进入。跳过数不进 manifest：否则勾选一张未检查的图片就会得到新的数据集标识，发布前要重新评估。它随模型目录发布，两次发布之间的数据变化（例如新增一个来源、某个类别的实例从 0 增加到上千）在审阅 `release.json` 的差异时直接可见。统计不放进 `sources.lock.json`：锁文件只在类别变化时改动，它出现在差异中就意味着下游需要确认（§15.1）。

nvinfer 配置是人工维护的运行参数，引用 `artifacts/model.onnx` 与 `artifacts/labels.txt`。首次发布前可以先写好（暂存目录中的校验连同新的 `release.json` 一起检查它）；还没有配置时发布照常进行并提示补写。之后发布只改动其中的 engine 路径。engine 路径的预检在评估与导出之前，路径写得无法识别时不会白等。

`release.json` 不保存权重本身，只记录权重的 sha256 与 run 标识：权重留在训练根目录，用于续训、预标注、退化比较与复现。删除一个已发布的 run 之后，下一次发布无法与它比较，只能 `--accept-metrics`。

## 9. 训练机与镜像

- 训练镜像（`framework/docker/train.Dockerfile`）基于 PyTorch 官方 CUDA 镜像加训练后端与 onnx/onnxruntime，版本固定，只在 x86 GPU 训练机运行。镜像只提供运行环境，训练工具（`framework/training/`）不复制进镜像：依赖（含间接依赖）全部固定在 `framework/docker/train-requirements.txt`；`scripts/train.sh` 以 Dockerfile 与依赖文件的摘要作为 tag，缺少时自动构建；以当前用户运行，挂载仓库（可写）、训练根目录（可写）与当前目录（只读），参数中含 `/` 的已有路径按真实路径只读挂载。
- Ultralytics 以 AGPL-3.0 发布：它只存在于训练镜像中，不进入 base、apps 或发布包；导出的 ONNX 是否受其许可约束，商用前需要单独确认。
- 不依赖 GPU 的部分（标注读取、去重、划分、manifest、规格与发布记录）是纯 Python，在 builder 的 CTest 中测试；训练、导出与一致性验证只在训练机上验证。

## 10. 打包与安装

- `scripts/build-release.sh` 在渲染之后、检查镜像之前执行模型校验（也可单独运行 `framework/deploy/dsf-deploy.py check-models`）。已实现的检查：规格完整；规格的类别、输入、输出布局与发布记录一致；产物与 sha256 一致且没有未发布的文件；`labels.txt` 与类别一致；导出签名符合输入尺寸与已知输出布局；nvinfer 配置引用的 ONNX 与标签是已发布产物，`model-engine-file` 在 `/data/engines/<id>/` 下，`num-detected-classes` 等于类别数，固定 batch 导出时 `batch-size` 与之相等。
- 服务的类别依赖（§15.1）：`services/<服务>/models.lock.json` 记录的各模型类别必须与模型的 `release.json` 相同，不同时打印差异（`dsf-deploy.py lock-models <服务> [<模型>…]` 看过差异后重写）；没有这个文件的服务不检查。发布包不带 `services/`，这项只在开发机打包时检查。
- 二级模型（§12）另外检查：`softmax` 输出为 `[batch, 类别数]`；`labels.txt` 为一行、以 `;` 分隔；nvinfer 配置的 `network-type=1`、`process-mode=2`，`operate-on-gie-id` 与 `operate-on-class-ids` 与一级模型的配置和发布对得上。
- 模型随发布包进入安装目录的 `models/`；`model-engine-builder` 在目标 GPU 上构建 engine，Jetson 与服务器各自构建，engine 不进包。

## 11. 任务扩展

任务适配器决定一个任务从标注到部署的每一步；新增任务只加适配器与对应的解析插件，不改流程：

| 任务 | 标注 | 后端视图（Ultralytics） | 指标 | nvinfer | 输出与解析 | 状态 |
|---|---|---|---|---|---|---|
| detect | 矩形框 | YOLO 检测格式 | mAP | `network-type=0` | `xyxy-confidence-class_id`，`plugins/yolo-parser` | 首先实现 |
| classify | 父类别框上的属性（二级，§12） | 按类别分目录的裁剪 | accuracy、宏平均与每类 P/R | `network-type=1` | `softmax`（导出时追加），nvinfer 内置分类解析；`labels.txt` 为一行、以 `;` 分隔 | 二级已实现；整图分类预留 |
| segment | 多边形 | YOLO 分割格式 | 框与掩码 mAP | `network-type=3` | 实例掩码，需要配套解析插件 | 预留 |
| obb、pose 等 | 旋转框、关键点 | 后端对应格式 | 后端对应指标 | 视解析插件 | 视解析插件 | 不在计划内 |

适配器接口（`framework/training` 中，任务与后端各一组）：

- 任务：从一份标注解读出样本（并报告不认识的形状与类别）；写出后端视图；指标的名称与比较方向；导出时追加的输出转换；`labels.txt` 的格式；发布时与打包时的契约检查。
- 后端：训练、评估、原生推理（预标注与一致性验证用）、导出 ONNX；声明支持的任务。

还没有实现的任务在 `train init` 与 `model.yaml` 校验时明确报告“任务尚未支持”，而不是走到训练才失败。

## 12. 一级与二级模型

二级模型作用于一级模型检测到的目标（例如在“标志牌”框内再分类是否遮挡），在 `model.yaml` 中声明（`train init <id> --task classify --class <值>… --parent <父模型> --parent-class <父类别>… --attribute <属性>` 生成骨架）：

```yaml
task: classify
classes: [否, 是]                         # 属性的取值，顺序即类别号
cascade:
  parent: road-assets                   # 一级（检测）模型的 id
  classes: [标志牌(绿底矩形), 标志牌(圆牌)]  # 父模型中触发本模型的类别，按名称写
input: {width: 128, height: 128}        # 正方形
output_layout: softmax
data:
  sources: [road-assets]                # 通常与父模型相同
  ignore_classes: [未知]                 # 不收入的属性值
  from_parent:
    attribute: 遮挡                      # 训练样本 = 父类别框的裁剪，类别 = 框上该属性的值
    min_size: 16                        # 短边小于它的框不收入（现场同样太小，无法可靠分类）
```

- **数据**：二级模型复用父模型的来源。X-AnyLabeling 的形状属性让一次标注同时得到一级框与二级标签；`prepare` 从父类别框裁剪出样本，划分沿用这些框所在图片的采集组，两级模型的测试集来自同一批图片。没有该属性的框、属性值在 `ignore_classes` 中的框、小于 `min_size` 的框不收入，在报告中按原因计数；其他属性值是错误。类别依赖（§15.1）检查的是来源中有 `cascade.classes`。训练划分中没有样本的类别报错：它无法学习。
- **裁剪与现场一致**：二级 nvinfer（`process-mode=2`）把检测框原样缩放到网络输入，不保持宽高比、不外扩。训练样本按同样的方式生成（拉伸到 `input`，写成 PNG），所以不提供外扩（`padding`）；nvinfer 配置用 `maintain-aspect-ratio=0`，`input-object-min-width/height` 与 `min_size` 一致。后端评估时的“缩放 + 中心裁剪”对正方形样本不起作用。
- **评估**：在真实框的裁剪上评估二级模型本身。类别常常极不均衡（“遮挡=是”只占百分之几），全猜多数类也有很高的 accuracy，所以指标是 accuracy、各类 recall 的平均（`recall`）与各类 precision 的平均（`precision`），每类给出 recall 与 precision，骨架的退化门槛比较 `recall`。训练时也按验证划分上的 `recall` 选择 `best.pt` 并判断早停（Ultralytics 默认按 top-1/top-5 的平均，二分类时就是 accuracy，会保留少数类召回差的检查点、过早停止），`results.csv` 中为 `metrics/recall_mean`。端到端评估（一级检测框 → 二级分类）以后按需要增加。
- **导出**：Ultralytics 的分类头输出 logits（softmax 只在它的 Python 预测器中做），而 nvinfer 内置的分类解析把输出当作概率与 `classifier-threshold` 比较，所以导出时追加 Softmax，输出为 `[batch, 类别数]`。一致性验证在测试划分的裁剪上比较概率。
- **部署契约**：二级模型的 nvinfer 配置使用 `network-type=1`、`process-mode=2`，`operate-on-gie-id` 等于父模型配置的 `gie-unique-id`，`operate-on-class-ids` 由 `cascade.classes` 按父模型发布的 `classes` 换算出类别号。打包与发布时校验：父模型重新发布并改变了类别号时，二级模型的配置在打包前失败，而不是在现场作用于错误的类别。
- 应用配置的 `inference` 链按顺序串接一级与二级模型（已支持），二级结果在检测消息对象的 `classes[]` 中。
- **尚未实现**：二级模型的预标注（`train prelabel` 对它报告不支持）；父类别改名或删除时自动改写 `cascade.classes`——`train update` 列出受影响的二级模型，由人改写，在改写之前二级模型的规格校验失败；整图分类。`train import` 与 `train update add|rename|retire` 对二级模型拒绝：它的样本来自父模型的来源，类别是属性值而不是标签。

## 13. 只更新模型

模型比程序更新得频繁。`scripts/build-model-package.sh <id>` 打出只含 `models/<id>/` 的模型包，目标机在其中运行 `./install.sh`：预检之后在旧服务继续运行时构建 engine（engine 路径带发布标识，§8），再切换模型目录并只重启引用模型的服务，停机时间从 TensorRT 构建的数分钟缩短到重启的数秒；上一份模型目录保留，`./install.sh --rollback` 切回并重启。不再被当前与上一份模型引用的 engine 由构建工具清理。细节见 [deployment.md](deployment.md) §7。

INT8：`prepare` 可从训练划分抽取校准图像，发布时生成校准缓存并作为产物发布（`int8-calib-file` 已计入 engine 指纹）。

数据闭环：现场应用已经发布检测结果并录像，可以增加一个服务按规则（低置信度、类别混淆、人工标记）抽取帧与预测结果，作为预标注导入新的来源，回到 §5.4。

## 14. 实现顺序

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 模型目录格式、`release.json`、打包前的模型校验 | 已实现 |
| 2 | 训练镜像、`scripts/train.sh`；`train init`、`train import`（图片去重、视频抽帧）；X-AnyLabeling 读取与 `train accept`；`train prepare`（检测） | 已实现 |
| 3 | `train run` / `train eval`（Ultralytics，检测）；分片评估除外 | 已实现 |
| 4 | `train prelabel` | 已实现 |
| 5 | `train publish`（检测：门槛、导出、一致性验证、按发布命名 engine） | 已实现 |
| 6 | 模型包与停服前构建 engine | 已实现 |
| 7 | 二级分类模型及其打包校验（§12） | 已实现（预标注与 `cascade.classes` 的自动改写除外） |
| 7b | 整图分类、分割任务 | 预留 |
| 8 | INT8 校准、数据闭环 | 预留 |
| 9 | 修改已有模型的类别：类别依赖锁文件、`train update`、按类别预标注（§15） | 已实现（二级模型的 `cascade.classes` 由人改写，§12） |

## 15. 修改类别

模型的类别会变：现场要求新增一种目标，两类分不清要合并，一类要细分，名称要改。目前只能手改 `model.yaml`，有两个问题：

- **新类别在旧标注里是背景。** 改之前已检查的图片里没有新类别的框，`prepare` 照常收入，训练与测试都把画面中的新类别目标当作背景：模型学着不去检测它，测试集也奖励这种行为。
- **改动的影响看不见。** 类别名写在每份标注、每个来源的 `labels.txt` 里，类别号被二级模型与下游服务使用；改了一处，其余的是否跟着改、是否还匹配，没有任何检查。

设计用两件事解决：下游记录确认时上游的类别列表，让影响可推导、可检查；撤销 `checked` 让需要重新看的图片回到“未确认”，沿用“只收人确认过的标注”这条已有规则。

### 15.1 类别依赖

来源、模型、服务三者的依赖是 **来源 → 模型 → 服务**。两段依赖只有一条规则：**下游记录确认时上游的类别列表（有序，顺序即类别号），与上游当前的列表不同就要求重新确认。** 不另设版本号：服务关心的就是类别名与顺序本身，只要列表没变（包括改了又改回），下游什么都不用做。

| 依赖 | 上游的类别列表 | 下游的记录（工具写入，随下游提交） | 不一致时 | 确认 |
|---|---|---|---|---|
| 来源 → 模型 | `source.json` 的 `classes`，即 `labels.txt` | `models/<id>/sources.lock.json`：每个来源的类别列表 | `prepare`、`run`、`publish` 拒绝 | `train update <id> --confirm` |
| 模型 → 服务 | `release.json` 的类别 | `services/<服务>/models.lock.json`：每个模型的类别列表 | 打包校验失败 | `framework/deploy/dsf-deploy.py lock-models <服务> [<模型>…]` |

- **谁解释类别，谁锁定。** 模型 → 服务的依赖是语义上的，不看服务在哪里加载模型：后处理、告警、按类别过滤或按类别号写规则的服务，即使只从消息中拿到检测结果，也锁定它依赖的模型；新增类别时打包失败，提醒它补上对新类别的处理。只加载模型推理、把类别原样透传的服务不锁定：它与模型的兼容由模型校验保证（`labels.txt`、`num-detected-classes`，§10），类别变化时它不需要修改，锁定它只会让确认落在不需要看差异的地方。
- **记录是事实，不是规格。** 锁文件由工具写入，不手改，与 `release.json` 一样；`model.yaml` 的 `data.sources` 仍是来源名列表，compose 不需要改动（部署工具只用标准库，不改写带注释的 YAML）。
- **报差异，不只报不一致。** 不一致时和确认时都按名称与位置打印差异（新增、删除、第 n 类由 A 改为 B），人据此判断下游是否需要修改；锁文件的变化在评审中同样直接可见。列表很短，直接保存列表而不是摘要，差异才能还原。
- **上游变了，下游即使不需要任何修改也要显式确认一次。** 确认就是在看过差异之后重写锁文件并提交。
- **模型的类别是来源类别的子集。** 模型的每个类别必须在它依赖的每个来源的类别中，否则这些来源的图片无法为它作证，`prepare` 拒绝。来源中有、模型不用的类别在读取时忽略（不再需要逐个写进 `data.ignore_classes`）；两者都没有的标签照常报错，拼写错误不会悄悄变成新类别。手改 `model.yaml` 新增或改名类别时因此无法 `prepare`，必须经 `train update`；手工删除一个类别等同于 `retire`（§15.3）。
- **二级模型** 的 `cascade.classes` 按名称引用父类别（§12），打包时按父模型的类别换算编号，不另设锁文件；父模型改名或删除了被引用的类别时打包失败。
- **迁移。** `source.json` 还没有 `classes` 的来源以 `labels.txt` 为类别（导入一向按模型的类别写出它），下一次导入或改写来源时记入 `source.json`。模型还没有 `sources.lock.json` 时，`prepare` 按来源当前的类别写出并提示，相当于首次确认；没有 `models.lock.json` 的服务不检查。仓库现有的数据因此不需要任何操作。

### 15.2 撤销与复核名单

新增类别时，来源中已检查的图片需要有人为新类别重新看一遍。工具把它们改为 `checked: false`，它们按“未检查”的规则退出数据集，人在 X-AnyLabeling 中补上新类别的框并重新勾选后回来。

- **先记录，再撤销。** `source.json` 的 `reviews` 先记下这次撤销：新类别、`--from` 的类别、时间、被撤销的文件，与来源的新类别在同一次写入中生效；然后才改标注文件，全部改完才把这份名单标为 `revoked: true`。撤销之后“以前检查过、只差新类别”与“从没标过”的图片从标注上分不出来，这份名单是按类别预标注（§15.4）与进度统计的依据。中断后重新执行同一命令，按未完成的名单继续。
- **报告进度。** `prepare` 把名单中仍未勾选的图片按“待复核 <类别>”计入跳过原因，与“未检查”分开，按来源列出；`train update <id>`（不带操作）列出每个来源每次复核的进度。
- **不能批量跳过。** `train accept` 拒绝名单中仍未勾选的文件，除非给出 `--reason`，理由与时间记入名单。
- 收录规则不变：名单只用于报告与预标注，进不进数据集仍然只看 `checked`。

### 15.3 操作：`train update <id> <操作>`

每次运行先完成全部检查（冲突、可写；起点必须是确认过的状态，其他模型带来的、恰好就是这次修改的来源变化也算确认过），再写入（`add` 的顺序见 §15.2）：标注文件 → 来源记录与 `labels.txt` → `sources.lock.json` → 其他模型的 `model.yaml` → 本模型的 `model.yaml`。标注先改：中断时来源记录还是旧名称，`prepare` 遇到新名称报未知类别，不会在不一致的状态下训练；重新执行同一命令即可完成，已改过的文件跳过。`model.yaml` 只改写类别列表中受影响的行，注释与其他内容不变。打印每个来源与模型的类别变化、撤销的图片数、自动修复的内容与需要人处理的依赖。改写来源期间不要在 X-AnyLabeling 中保存这个来源的图片；完成后在工具中重新打开目录并重新导入 `labels.txt`。

| 操作 | 本模型 | 来源 | 其他模型与服务 |
|---|---|---|---|
| `add <名称>` | 追加在 `classes` 末尾 | 已有这个类别的来源不变（之前已为它标注过）；其余来源追加类别、刷新 `labels.txt`，撤销已检查的图片并记入名单（§15.2） | 依赖这些来源的其他模型锁文件不一致，不可用，复核后 `--confirm`；本模型的服务需要确认 |
| `add <名称> --from <类别A>` | 同上 | 同上，但只撤销含 A 框的图片：新类别的目标以前被标成 A（细分 A） | 同上 |
| `add … --skip-source <来源> --reason <文本>` | — | 该来源追加类别但不撤销，理由记入名单：确定这批素材不含新类别 | — |
| `rename <旧> <新>` | 原位改名 | 改写标注中的名称（`checked` 不变），来源类别改名 | **自动修复**：依赖这些来源且含旧名称的模型同样改名；不含的模型类别不受影响，自动更新锁文件；以本模型为父的二级模型改写 `cascade.classes`（尚未实现：列出受影响的二级模型，由人改写）；所有改了名的模型的服务需要确认 |
| `retire <名称>` | 从 `classes` 删除；后面的类别号前移 | 不变：其他模型可能还在用这个类别，本模型读取时忽略它 | 二级模型引用了它时**拒绝**，提示先调整二级模型（尚未实现：现在只列出受影响的二级模型）；服务需要确认 |
| `retire <A> --into <B>` | 删除 A（合并进 B） | 改写标注中的 A 为 B（`checked` 不变），来源类别删除 A | 依赖这些来源的其他模型仍把 A 当作独立类别时**拒绝**，提示对那个模型先做同样的合并，或改为只 `retire`；二级模型的 `cascade.classes` 中的 A 改为 B（尚未实现：列出，由人改写） |
| `--confirm` | 按来源当前的类别重写 `sources.lock.json` | — | — |
| （不带操作） | 列出类别、与各来源锁文件的差异与复核进度 | — | — |

- 删除会改变后面每个类别的编号。这是安全的：二级模型的 `operate-on-class-ids` 在打包时按名称换算（§12），类别列表变了，服务必须重新确认。前提是下游按名称（检测消息的 `label`）而不是按编号识别类别；确认时检查这一点。
- 不支持调换顺序；确实需要时手改 `model.yaml` 并按不兼容的变更处理。
- 新名称不能与任何相关来源中已有的类别重名。

### 15.4 按类别预标注：`train prelabel <id> --source <来源> --class <类别> [--run <run>]`

- 只处理该来源名单中为这个类别撤销、仍未勾选的图片：在已有标注上追加这个类别的候选框，已有的框一律不动，图片保持 `checked: false`。与已有同类别框 IoU ≥ 0.5 的候选不写，重复运行不会叠加。
- 这是 §5.4“永不覆盖已有标注”的唯一例外，范围限于名单中未勾选的文件。读取与写回之间比对文件的 sha256，人在工具中保存过的图片跳过。
- 所用模型只需认识这个类别（按名称），类别列表不必与规格相同；通常是用重新勾选的图片训练出的 v0（§15.5）。
- 不带 `--class` 时的类别检查同时放宽为：模型的类别都在规格中（按名称），写出它认识的类别。新增类别之后，当前发布的模型仍可为新素材预标注旧类别，人只补新类别。
- 按类别 NMS（不让别的类别的高分框压掉新类别的候选）。写入的文件与所用模型记入 `prelabel.json` 的 `class_images`，`prepare` 照常统计保留、删除与补充的框数（§5.4），只比较这个类别的框。

### 15.5 新增类别的流程

```text
train update road-assets add 防撞墩           来源与模型追加类别，撤销已检查的图片
（在已有来源中挑几十张含防撞墩的图，补框后勾选；可用 X-AnyLabeling 内置的开放词汇模型辅助寻找）
train prepare road-assets && train run road-assets
                                              v0：只有重新勾选的图片进入数据集
train prelabel road-assets --source batch-01 --class 防撞墩 --run <v0>
（复核、勾选；数据多了可以再训练一版，用更好的模型预标注剩下的来源）
……每个来源的名单都处理完
train prepare road-assets && train run road-assets && train publish road-assets <run>
（修复其他模型的依赖：train update <其他模型> --confirm；dsf-deploy.py lock-models <服务>）
```

- v0 只用来给新类别出候选框。训练数据很少，起点权重用当前发布的 run（`train.weights: run:<run>`）：类别数变化时 Ultralytics 只加载形状相同的层，检测头重新初始化，其余特征继承。
- 重新勾选的图片包含全部类别，v0 是普通的完整训练，不需要特殊的数据处理。
- 名单还没处理完时也可以 `publish`：未勾选的图片不在数据集中，不会把新类别当作背景，门槛照常检查。`publish` 列出未完成的复核，并随按来源统计写入 `release.json`（§8）。

### 15.6 评估与发布

- **退化比较在共有类别上做。** 按名称把当前发布的类别对应到新类别，改名按 `sources.lock.json` 的 `renamed`（`train update rename` 写入当前发布以来的改名，发布后清空）对应。当前模型在新数据集的测试划分上评估时，对象换成它的类别号、它不认识的类别去掉；比较的是两个模型在共有类别上每类指标的平均（只算测试划分中有实例的类别），结果写入门槛的 `classes`、`excluded` 与 `renamed`；`--from A` 的 A 不计入（旧模型会把新类别的目标报成 A，比较对它不公平）。
- `release.json` 记录相对上一个发布的类别变化（`class_changes`：改名、删除、新增），以及按来源的标注统计与未完成的复核（`training_data`，§8），审阅差异时直接可见。
- 打包校验在现有检查（§10）之外增加：服务的 `models.lock.json` 与模型发布的类别一致，不一致时打印差异。
- 模型包仍然拒绝类别变化（§13），需要完整发布包：服务的依赖确认随完整发布包一起交付。

### 15.7 实现顺序

1. `sources.lock.json` 与迁移；`prepare`、`run`、`publish` 的依赖检查、差异报告与 `--confirm`；读取时忽略来源中本模型不用的类别。**已实现**
2. `rename`、`retire`（含 `--into`）及其自动修复与冲突检查。**已实现**；作用于父类别的二级模型由 `train update` 列出，`cascade.classes` 由人改写（自动改写尚未实现），类别号的变化由打包时的级联校验发现（§12）。
3. `add`：撤销与复核名单、`--from`、`--skip-source`；`prepare` 的复核进度；`accept` 的限制。**已实现**
4. `prelabel --class` 与放宽的类别检查。**已实现**
5. `lock-models` 与服务的打包校验；共有类别上的退化比较；`release.json` 中的类别变化。**已实现**

除训练与预标注推理外都是纯 Python，可以在 builder 中测试。

### 15.8 待决定

- **服务是否必须有 `models.lock.json`。** 先作为可选：有了才检查。解释类别的服务都应该锁定（§15.1），只透传类别的服务不需要，所以不能简单地改为“引用模型的服务必须锁定”。
- **只比较名称的锁。** 按名称识别类别的服务不关心顺序，`retire` 使后面的类别号前移时它们也要确认一次，通常不需要修改。可以给锁增加只比较名称的模式；`retire` 不常发生，先统一比较有序列表，对按类别号识别的服务更安全。
- **`retire` 是否提供 `--purge`** 删除来源中这个类别的框。没有其他模型使用时可以清理来源；先不提供，来源中多余的类别在读取时忽略，不影响训练。
- **模型包能否承载类别变化。** 目标机上已安装服务的锁文件与新模型的类别一致时，理论上可以只更新模型；等出现频繁的类别调整时再放宽。
