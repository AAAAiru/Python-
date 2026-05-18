# 抑郁症倾向识别（英文社交媒体文本）课程设计工程

本项目使用 **Python 3.11+**（推荐用仓库根目录 `.venv`，见下文），面向 Kaggle 英文心理健康文本数据，完成 **二分类：Depression vs 其他**（默认将 *Anxiety / Normal / Suicidal* 等均视为负类，可在 `src/depression_ml/config.py` 中调整 `POSITIVE_LABELS`）。

> **重要声明**：仅供课程与科研教学演示，不能作为医学诊断或危机干预依据。涉及自伤风险时请引导用户联系当地急救或心理援助热线。

## 目录结构

- `data/`：放置从 Kaggle 下载的 CSV（文件名见下）。
- `src/depression_ml/`：可复用库代码（数据加载、特征、训练、评估、风险分层）。
- `scripts/run_train.py`：命令行训练入口。
- `scripts/run_gui.py`：训练完成后启动 Tkinter 图形界面。
- `notebooks/depression_detection.ipynb`：Jupyter 全流程演示。
- `artifacts/`：训练产物（模型、向量化器、指标、图像）。

## 1. 环境

在项目根目录下执行：

```bash
# macOS / Linux：用本机已安装的 Python 3.11+ 创建虚拟环境（示例为 python3.12）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Windows 将 `source .venv/bin/activate` 换为 `.venv\Scripts\activate`。

或：

```bash
pip install -r requirements.txt
```

并把 `src` 加入 `PYTHONPATH`（脚本已自动处理）。

若使用 **python.org 安装的 Python 3.12**，首次 `pip install` 出现 `SSL: CERTIFICATE_VERIFY_FAILED`，请先运行一次 **「Install Certificates.command」**（在 `/Applications/Python 3.12/` 文件夹内），或临时使用：

`pip install ... --trusted-host pypi.org --trusted-host files.pythonhosted.org`

## 2. 数据集（任选其一）

### A. Mental Health Text Classification Dataset（推荐）

Kaggle：<https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset>

建议将以下文件放入 `data/`（文件名以你下载的压缩包为准，常见命名如下）：

- 训练：`mental_heath_unbanlanced.csv`（部分镜像拼写为 *heath* / *unbanlanced*）或 `mental_health_combined_train.csv`
- 测试：`mental_health_combined_test.csv`（约 992 条平衡测试集）

数据列通常为：`text` + `status`（四分类：Suicidal, Depression, Anxiety, Normal）。

### B. Sentiment & Mental Health Dataset (Reddit-Based)

Kaggle：<https://www.kaggle.com/datasets/maazkareem/sentiment-and-mental-health-dataset-reddit-based>

将 CSV 放入 `data/`（常见列为 `statement` 与 `status`）。默认脚本在检测到该单文件布局时，会 **仅保留 Depression 与 Normal** 做二分类（可在 `config.py` 修改 `REDDIT_BINARY_FILTER`）。

### C. 无数据时的占位集

若 `data/` 下没有任何 CSV，训练脚本会自动生成 `synthetic_train.csv` / `synthetic_test.csv` 以便连通性自检（指标无科研意义，仅用于验收管道）。

## 3. 训练

```bash
python scripts/run_train.py
```

可选参数：

- `--slow`：对部分模型启用 `GridSearchCV`（更慢）。
- `--no-oversample`：关闭训练集随机过采样。
- `--data-dir`、`--artifacts-dir`：自定义路径。

训练结束后查看 `artifacts/metrics.json` 与各类 `*.png` 曲线图。

## 4. Jupyter

```bash
jupyter notebook notebooks/depression_detection.ipynb
```

## 5. GUI

```bash
python scripts/run_gui.py
```

界面会校验**最短字符数**与**是否主要为英文**；结果区除模型概率外，会展示 **EMNLP’17 `user_selection` 词典** 的命中摘要（解释用）。若训练阶段拟合了 **Platt 概率校准**，推理概率为校准后分数（见 `artifacts/platt_calibrator.pkl`）。

### 5.1 批量推理（CSV）

```bash
python scripts/run_infer_csv.py --input data/depression_dataset_reddit_cleaned.csv --output artifacts/predictions_sample.csv
```

可用 `--text-column` 指定文本列；默认自动匹配 `clean_text` / `text` 等常见列名。

### 5.2 单元测试

```bash
pip install -e ".[dev]"
pytest
```

## 6. 任务与模型

- 预处理：英文小写、去 URL、非字母清洗（`preprocess.py`）；可选 **拉丁字母占比** 启发式（`looks_english`，阈值见 `config.py`）。
- 特征：`TF-IDF`（稀疏矩阵）+ 词长/词数等统计量 + **VADER** 情感复合分 + **Georgetown `emnlp17-depression` 仓库中 `user_selection/`** 资源：`mh_patterns.txt`、`mh_subreddits.txt`（子版块名 / `r/…` 风格提及）、`diagpatterns_*` + `expansions.json`（`emnlp17_signals.py` / `features.py`），`StandardScaler(with_mean=False)`。参考代码里的 **Keras 用户级 CNN**（`reference/.../model/`）依赖过旧栈，本工程未嵌入，仅用其 **词典侧** 弱特征。
- 模型：逻辑回归、`LinearSVC`（`CalibratedClassifierCV` 概率）、随机森林、**XGBoost**（若安装失败则自动跳过）。
- 大规模训练集：默认对 **训练子集** 做分层抽样至 `MAX_TRAIN_ROWS`（见 `config.py`），验证集与官方测试集仍全量用于评估，以控制笔记本/普通电脑的内存占用。
- 不平衡：`RandomOverSampler` + 多数模型的 `class_weight`。
- 验证集上按 **F2**（偏重召回）搜索分类阈值，并写入 `artifacts/risk_thresholds.json` 供 GUI 三档风险展示；可选在验证集上拟合 **Platt 校准**（`probability_calibrate.py`），并输出可靠性图 `artifacts/calibration_reliability_test.png`。
- 数据说明与伦理提示会写入 `artifacts/dataset_meta.json` 的 `label_disclaimer` 字段。

## 7. 许可与伦理

请遵守各数据集作者在 Kaggle 页面上的使用条款；课堂展示需写明数据来源、局限性与伦理风险（误判、污名化、隐私等）。
