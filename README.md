# 混凝土裂缝图像筛查器 (Concrete Crack Image Screener)

> D2 作业：为维护人员制作一个初步图像筛查器，对比多数类基线、全连接网络、小型 CNN 与预训练模型微调，重点检查漏检裂缝。

## 问题与使用者

设施维护团队会收到大量混凝土表面照片。人工逐张查看需要时间，而照片中的光照、纹理、阴影和污渍又可能与裂缝相似。团队希望先用程序把照片分成"可能有裂缝"和"未发现裂缝"两组，把可能有裂缝的照片优先交给人员复核。

**使用者**：需要先筛选大量照片、再安排人工复核的设施维护团队。

**产品边界**：这是一个初筛工具，不是结构安全鉴定工具。无论模型结果如何，最终判断仍由现场人员或工程师完成。

## 真实数据

**数据集**：Kaggle Concrete Crack Images (Surface Crack Detection)
- **URL**：https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- **规模**：Positive 与 Negative 各 20,000 张真实图像（共 40,000 张）
- **图像格式**：227x227 像素，RGB，JPEG
- **许可**：CC0: Public Domain

### 下载数据

**方法一：使用 Kaggle CLI（推荐）**

```bash
pip install kaggle
# 将 kaggle.json 放到 ~/.kaggle/ (从 https://www.kaggle.com/settings -> API -> Create New Token 下载)
python scripts/download_data.py
```

**方法二：手动下载**

1. 访问 https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
2. 下载 ZIP 文件
3. 解压到 `data/raw/`，确保有以下结构：
   ```
   data/
   └── raw/
       ├── Positive/    # 20,000 张有裂缝图像
       └── Negative/    # 20,000 张无裂缝图像
   ```

### 生成测试数据（仅供代码验证）

如果不方便下载数据，可以用合成数据验证代码是否能运行：

```bash
python scripts/make_test_data.py --count 80
```

> 注意：合成数据仅用于测试代码运行，不能用于实际结果。真实结果必须使用 Kaggle 数据集。

## 环境

### 系统要求

- Python 3.10+
- PyTorch 2.0+（CPU 版本即可）
- 约 2GB 磁盘空间（数据集 + 输出）

### 安装依赖

```bash
pip install -r requirements.txt
```

如果安装 PyTorch 遇到路径过长问题（Windows），请启用长路径支持或使用 CPU 版本：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pillow numpy
```

## 运行命令

### 1. 完整流程（数据检查 → 基线 → 超参数搜索 → 训练4模型 → 评估 → 比较）

```bash
python -m src.main
```

可选参数：
```bash
python -m src.main --epochs 10          # 指定训练轮数
python -m src.main --no-hp              # 跳过超参数搜索（快速运行）
python -m src.main --check-only         # 仅检查数据
```

### 2. 仅检查数据

```bash
python -m src.main --check-only
```

预期输出：
```
DATA CHECK
  Positive: exists=True, images=20000
  Negative: exists=True, images=20000
  Total images: 40000

IMAGE READABILITY CHECK
  Positive: readable=True, checked=10, errors=0
    sample sizes: [(227, 227), (227, 227), (227, 227)]
    sample modes: ['RGB', 'RGB', 'RGB']

TENSOR SHAPE CHECK
  Positive sample tensor shape: torch.Size([3, 227, 227])
    dtype: torch.float32, min: 0.0000, max: 1.0000
```

### 3. 运行测试

```bash
python -m unittest tests.test_basic -v
```

预期输出：
```
Ran 32 tests in XX.Xs
OK
```

## 预期输出

运行完整流程后，程序会：

1. **打印数据检查结果**：确认两个类别各 20,000 张图像，检查图像可读性和张量形状
2. **创建固定数据划分**：从真实数据中取每类 4,000 张子集，划分训练集（每类 3,000 张）和测试集（每类 1,000 张）
3. **检查数据泄漏**：使用感知哈希检测训练集和测试集中的近似重复图像
4. **训练多数类基线**：计算训练集中数量最多的类别
5. **超参数搜索**：对 SmallCNN 网格搜索学习率 [0.0001, 0.001, 0.01] × 训练轮数 [3, 5, 10]
6. **训练 SmallCNN**：使用最佳超参数在训练集上训练
7. **训练 FCNet**：全连接网络对比基线（无卷积层，39.6M 参数）
8. **训练 Pretrained ResNet18**：冻结骨干网络，仅微调 FC 层（1,026 可训练参数）
9. **评估并比较**：在相同测试集上比较 4 个模型的准确率、精确率、召回率、F1
10. **分析错误**：列出被漏检的裂缝（False Negatives）和误报（False Positives）
11. **保存结果**：
    - `outputs/results/results.json` — 完整结果（4 模型对比 + 超参数搜索）
    - `outputs/figures/training_history_cnn.png` — SmallCNN 训练曲线
    - `outputs/figures/training_history_fc.png` — FCNet 训练曲线
    - `outputs/figures/training_history_pretrained.png` — ResNet18 训练曲线
    - `outputs/figures/confusion_matrices.png` — 4 模型混淆矩阵对比
    - `outputs/figures/error_cases.png` — 错误案例图像
    - `outputs/models/*.pth` — 所有模型权重

## 项目结构

```
.
├── README.md                   # 本文件
├── report.md                   # 书面报告
├── presentation.pptx           # 答辩幻灯片
├── submission.json             # 提交清单
├── requirements.txt            # Python 依赖
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py               # 配置常量
│   ├── data_check.py           # 数据检查
│   ├── dataset.py              # 数据集加载与划分
│   ├── baseline.py             # 多数类基线
│   ├── model.py                # SmallCNN 架构
│   ├── fc_model.py             # FCNet 全连接网络（对比基线）
│   ├── pretrained.py           # 预训练 ResNet18 微调
│   ├── hyperparam_search.py    # 超参数网格搜索
│   ├── train.py                # 训练脚本（支持 4 种模型）
│   ├── evaluate.py             # 评估与错误分析
│   ├── leakage_check.py        # 数据泄漏检测
│   └── main.py                 # 主流程入口（12 步 pipeline）
├── tests/
│   ├── __init__.py
│   └── test_basic.py           # 单元测试
├── scripts/
│   ├── download_data.py        # Kaggle 数据下载
│   ├── make_test_data.py       # 合成测试数据生成
│   └── make_presentation.py    # PPT 生成脚本
└── outputs/                    # 运行输出（自动创建）
    ├── results/
    ├── figures/
    └── models/
```

## 关键设计决策

### 基线：多数类预测
基线总是预测训练集中数量最多的类别。它不读取图像内容，用于证明 CNN 是否真的比"总猜一种类别"更有用。在平衡数据集中（Positive 和 Negative 各 20,000 张），基线准确率约 50%。

### 对比基线：FCNet（全连接网络）
三层全连接网络：Flatten → Linear(155529→256) → Linear(256→128) → Linear(128→2)。无卷积层，直接展平像素作为输入，参数量 39.6M。用于证明卷积结构对图像任务的优势。

### 候选：SmallCNN
三层卷积网络：Conv(3→16→32→64) + MaxPool + FC(128→2)。参数量 6.4M，通过卷积权重共享和局部感受野有效提取裂缝的空间特征。

### 性能突破：Pretrained ResNet18
使用 ImageNet 预训练的 ResNet18，冻结骨干网络，仅微调最后的 FC 层（1,026 可训练参数）。利用迁移学习在有限数据下实现性能突破。

### 超参数搜索
对 SmallCNN 进行网格搜索：学习率 [0.0001, 0.001, 0.01] × 训练轮数 [3, 5, 10]，在验证集上选择最佳配置。

### 最重要的指标：裂缝召回率
裂缝召回率 = 正确检出的裂缝 / 所有真实裂缝。漏检裂缝（False Negative）是最重要的错误——它会使需要人工查看的照片没有被优先发现。

### 数据泄漏检查
使用感知哈希（average hash）检测训练集和测试集中是否存在近似重复图像。如果相似图像被分到训练和测试两边，测试成绩会过于乐观。

## 限制

1. **不是安全鉴定工具**：输出仅用于安排人工复核，不能替代现场检查、工程师判断或安全决策
2. **数据依赖**：结果完全取决于训练数据的质量和覆盖范围
3. **固定划分**：使用固定随机种子划分，结果可复现但可能因划分不同而变化
4. **合成数据局限**：合成数据仅用于验证代码可运行性，真实结果必须使用 Kaggle 数据集
