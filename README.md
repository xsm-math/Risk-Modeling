# Credit Risk Modeling — Probability Estimation and Decision Costs

[![tests](https://github.com/xsm-math/Risk-Modeling/actions/workflows/tests.yml/badge.svg)](https://github.com/xsm-math/Risk-Modeling/actions/workflows/tests.yml)

**真实公开数据上的信用卡违约研究：风险排序、概率校准与成本敏感决策。**

[完整研究报告](reports/credit/REPORT.md) · [实验协议](docs/credit_protocol.md) · [数据字典](docs/credit_data_dictionary.md) · [验证记录](docs/credit_validation.md)

## 研究设计

使用 UCI Default of Credit Card Clients 的 **30,000条历史客户记录**，研究观察期结束后的下一月违约风险。对照线性逻辑回归、WOE逻辑回归、直方图梯度提升树及其独立sigmoid校准版本。模型选择依据为验证集 Log loss；策略阈值也仅使用验证集选择，最终测试集不参与选型。

19个账户行为字段加5个固定衍生特征；不使用ID或人口属性。按相同账户输入分组，避免重复特征记录跨集合；约60%训练、10%校准、10%策略验证、20%最终测试。数据只有一个历史观察期，因此不声称进行了OOT验证。

## 测试结果

以下为固定配置、固定分组划分的结果；全部候选和常数概率基准见[结果表](reports/credit/test_metrics.csv)。

| 模型 | ROC AUC ↑ | KS ↑ | Average precision ↑ | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|
| 线性逻辑回归 | 0.7662 | 0.4366 | 0.5185 | 0.1383 | 0.4402 |
| WOE逻辑回归 | 0.7774 | 0.4308 | 0.5478 | 0.1343 | 0.4298 |
| **梯度提升树（验证集选中）** | **0.7924** | **0.4519** | **0.5631** | **0.1322** | **0.4219** |
| 梯度提升树＋sigmoid | 0.7924 | 0.4519 | 0.5631 | 0.1323 | 0.4222 |

- 主模型AUC的分组bootstrap 95%区间为 **[0.7771, 0.8063]**。
- 相对WOE基准的AUC差为 **0.0150**，配对区间 **[0.0084, 0.0216]**。
- 此次sigmoid校准没有改善主模型的验证集或测试集概率损失，保留未校准模型。

假设误拒非违约者成本为1、放行违约者成本为5，验证集阈值为 **0.166414**。测试通过率 **54.35%**，通过群体违约率 **9.17%**，违约捕获率 **77.45%**。每客户假设成本为 **0.5345**，WOE基准为 **0.5600**（各自使用验证集选出的阈值）。差值区间 **[0.0060, 0.0453]**，仅反映固定模型下的测试抽样不确定性。

这些是历史客户上的离线结果；成本为归一化情景单位，不是银行实际收益。提高违约捕获率会牺牲通过率，不能只呈现单一提升数字。

![模型、概率与决策比较](reports/credit/benchmark.svg)

## 复现

Python 3.11或3.12：

```bash
git clone https://github.com/xsm-math/Risk-Modeling.git
cd Risk-Modeling
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-credit.txt
python -m src.credit download
python -m src.credit run
```

下载约5.3MB的UCI原始压缩包并核对固定SHA256；下载失败不会使用模拟数据替代。完整流程包含500次分组bootstrap。精确依赖版本见 [requirements-credit-repro.txt](requirements-credit-repro.txt)。

默认命令会重新生成 `reports/credit/` 中的结果。需要保留已提交报告时：

```bash
python -m src.credit run --output reports/generated
```

批量预测（输入需包含[数据字典](docs/credit_data_dictionary.md)中的19个账户字段，无需标签）：

```bash
python -m src.credit score --input customers.csv --output predictions.csv
```

模型在完整实验后生成于 `artifacts/credit_model.joblib`。输出包括违约概率与假设阈值下的拒绝标志。仅加载自己生成或信任的joblib文件；此接口是本地批量推理，不是线上服务。

测试：

```bash
python -m pip install -r requirements-dev.txt -r requirements-credit.txt
python -m pytest -q
```

## 结构

| 路径 | 内容 |
|---|---|
| `src/credit/data.py` | 校验下载、字段校验、行内特征及分组划分 |
| `src/credit/models.py` | 三类基模型与独立校准 |
| `src/credit/evaluation.py` | 概率/排序指标、同分阈值搜索、配对分组bootstrap |
| `src/credit/experiment.py` | 冻结选型、策略比较、解释、图表、模型导出 |
| `configs/credit.json` | 预设模型容量、成本情景和随机种子 |
| `reports/credit/` | 聚合指标、成本敏感性、可靠性曲线、研究报告和运行清单 |
| `tests/` | 原评分卡与新真实数据流程的离线测试 |
| `src/risk/`、`src/train.py` | 原NumPy模拟评分卡实验 |

原始数据、逐客预测、拆分ID和模型文件默认保留本地，不提交Git。报告中的数据源、配置与拆分哈希用于追溯，不代替独立外部验证。

## 边界与延伸

本研究不是新客审批模型。UCI样本来自2005年的既有客户，不包含拒绝客户、实际违约损失、回收率或多期观测时点。研究未实现拒绝推断、真正的时间外验证和生产监控。去掉人口属性不代表消除了代理偏差。下一阶段应取得近期多期数据并验证特征可用时点，而不是针对已公开测试集反复调参。

原模拟研究独立保留：[模拟评分卡说明](SYNTHETIC_BASELINE.md)、[历史方法与限制](docs/methodology.md)，运行 `python -m src.train`。它的AUC和假设成本不能与新数据结果混为一组。

## 数据来源与参考

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H)，数据许可 **CC BY 4.0**。

实现参考：[scikit-learn概率校准](https://scikit-learn.org/stable/modules/calibration.html)、[直方图梯度提升](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)。本项目不主张新算法。
