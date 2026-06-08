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
pip install -e ".[train]"
```

Windows 将 `source .venv/bin/activate` 换为 `.venv\Scripts\activate`。

完整实验（MiniLM + Notebook）：

```bash
pip install -e ".[full]"
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

### D. 本地数据库（推荐）

按 `data/sources.json` **合并多源 CSV** 构建 SQLite（`data/depression.db`）及固定 **train / val / test** 划分：

```bash
python scripts/build_dataset.py --list-sources
python scripts/build_dataset.py --all-sources
```

可选 Kaggle 数据（下载后放入 `data/`，见 `data/README.md`）：

- `combined_data.csv` — *sentiment-analysis-for-mental-health*
- `urdu_depression_dataset.csv` — *urdu-depression-severity-dataset-2024-2025*

仅 Reddit 单文件：

```bash
python scripts/build_dataset.py --source data/depression_dataset_reddit_cleaned.csv
```

生成：`depression.db`、`train|val|test.csv`、`dataset_stats.json`（含各 `source_id` 行数）。训练优先读库。

## 3. 训练

快速训练（默认，只运行适合普通笔记本电脑的核心模型）：

```bash
python scripts/run_train.py --experiment quick
```

完整课程实验（模型比较、消融、MiniLM、跨数据源评估）：

```bash
python scripts/run_train.py --experiment full
```

可选参数：

- `--experiment {quick,full}`：轻量演示或完整实验。
- `--split-strategy {source_label,label}`：重建数据时的分层策略。
- `--seed`：统一控制数据切分和模型随机性。
- `--no-oversample`：关闭训练集随机过采样。
- `--data-dir`、`--artifacts-dir`：自定义路径。

训练结束后查看：

- `metrics.json`：验证集、同域测试、跨域测试和校准结果。
- `model_metadata.json`：代码版本、依赖、随机种子、数据指纹和训练时间。
- `ablation_results.csv`：完整实验的特征消融。
- `error_cases_test.csv`：高置信度误判样本，供课程报告错误分析。

## 3.1 特征消融与划分对比

对比 **TF-IDF 单独 / +VADER / +EMNLP 全特征**，并在存在多数据源（`depression.db` 含 2 个以上 `source_id`）时比较：

- **stratified_row**：按行分层随机划分（与默认建库一致）
- **group_by_source**：按数据源整组划分，减轻同源泄漏

```bash
python scripts/run_ablation.py
```

输出：`artifacts/ablation_report.json`、`ablation_table.csv`、`ablation_test_f2.png`（及可选 `ablation_split_protocol_f2.png`）。

## 3.2 任务 2：嵌入基线 + 跨域（OOV）评估

在同一 **train/val/test** 上对比：

- **TF-IDF 全特征** + 校准 `LinearSVC`
- **MiniLM 句向量** + 逻辑回归（`sentence-transformers`）

并在 `depression.db` 存在多源时做 **OOV**：默认仅在 **reddit** 上训练，在 `depression_text_clf` / `sentiment` / `urdu` 等源上测试泛化。

```bash
pip install sentence-transformers   # 若尚未安装
python scripts/run_task2.py
```

**国内访问 Hugging Face 超时（WinError 10060）** 时任选其一：

```powershell
# 方法 A：镜像（推荐，当前终端有效）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
python -u scripts/run_task2.py

# 或一键脚本
.\scripts\run_task2_mirror.ps1
```

```powershell
# 方法 B：先离线下载到 models/all-MiniLM-L6-v2，再跑 task2
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/download_embedding_model.py
python -u scripts/run_task2.py
```

```powershell
# 方法 C：暂时不做 embedding，仍可完成 TF-IDF + OOV
python -u scripts/run_task2.py --no-embeddings
```

可选：

- `--in-domain-only` / `--no-oov`：只做同分布对比，更快  
- `--no-embeddings`：跳过句向量（仅 TF-IDF）

输出：

| 文件 | 含义 |
|------|------|
| `artifacts/model_compare_table.csv` | 同分布 Test F2 / 召回对比 |
| `artifacts/model_compare_test_f2.png` | 柱状图 |
| `artifacts/oov_metrics.json` | 跨源详细指标 |
| `artifacts/oov_compare_table.csv` | 各 holdout 源 × 模型 |
| `artifacts/task2_summary.json` | 汇总 |

单独跑 OOV：`python scripts/run_oov_eval.py`

## 3.3 GUI（短句 / 积极句防误判）

**推荐（只需打开一次，不用每次测都开终端）：**

- Windows：双击项目根目录 **`启动抑郁症筛查Demo.bat`**
- 或命令行：`python scripts/run_gui.py`

**演示流程：** 粘贴文本 → **Run assessment** → 测下一条时点结果区 **「一键归零」**（`Esc` 快捷键），无需关掉窗口或重跑终端。

- 极短文本会提示「置信度低」，并在无心理词典命中时**限制最高档位**。
- 明显积极表述（VADER、积极词、词典）且分数较低时，**下调至低风险**（界面会注明规则说明）。
- 若模型档位与展示档位不一致，会显示 `model-only tier was …`。

## 4. Jupyter

```bash
jupyter notebook notebooks/depression_detection.ipynb
```

## 5. GUI

```bash
python scripts/run_gui.py
```

界面会校验**最短字符数**与**是否主要为英文**；结果展示的是“模型阳性分数”，不是患病概率或诊断。EMNLP’17 词典命中只作为文本证据展示，不应表述为模型的因果解释。

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
- 验证集被拆成互不重叠的校准子集和阈值/模型选择子集；Platt 校准与 F2 阈值不会拟合在同一批样本上。
- 完整实验按 `0.7 × 同域验证 F2 + 0.3 × 跨域 F2` 选择有跨域结果的模型。
- 报告 Accuracy、Precision、Recall、Specificity、F1、F2、MCC、ROC-AUC、PR-AUC、Brier score 与 bootstrap 95% 区间。
- 数据说明与伦理提示会写入 `artifacts/dataset_meta.json` 的 `label_disclaimer` 字段。

## 7. 许可与伦理

请遵守各数据集作者在 Kaggle 页面上的使用条款；课堂展示需写明数据来源、局限性与伦理风险（误判、污名化、隐私等）。

课程论文建议结构见 [`docs/course_report_outline.md`](docs/course_report_outline.md)。
