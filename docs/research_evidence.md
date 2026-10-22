# 信用风险研究：问题、证据与复现边界

本说明核对基线提交 `a11dbea51059320031e88a88a4d816a280cccc63` 的实际实现与已发布结果。
本次已实际完成完整复现、负对照与发布表对账，70 项测试通过；
执行记录及全部低位差异见 [2026-10-22 复现记录](../reports/audit/2026-10-22/README.md)。
下面数字来自仓库表格，不是新实验，也不是银行生产效果。原始合成实验位于
`src/risk/` 和 `outputs/`；这里仅讨论 `src/credit/` 与 `reports/credit/` 的真实公开数据研究。

## 1. 可检验的问题定义

在六个月账单与还款记录均已可用的假设下，预测已有信用卡客户下月违约标签。
比较可解释 WOE 评分卡与非线性树模型的排序、概率质量及固定非对称成本下的决策。
单位是一条客户历史，标签 1 表示违约；不是新客授信、欺诈检测或多期累计 PD。
没有逐字段入库时间，因此窗口结束后的可用性是研究假设，未验证真实系统延迟。

成功标准分开看：ROC-AUC/KS/AP 衡量排序；log loss/Brier 与可靠性分箱衡量概率质量；
阈值下的混淆矩阵、通过率和模拟成本衡量决策。模型由验证集 log loss 选定，
不是挑测试集最大 AUC，也不是直接按最低测试成本选模型。

## 2. 数据和特征实际做了什么

UCI Default of Credit Card Clients，30,000 行、6,636 个违约、违约率 22.12%。
2005 年 4–9 月历史预测下月标签；金额为新台币，但目标不是金额损失。
下载入口与 SHA256 固定在 [`data.py`](../src/credit/data.py)，无合成回退。
数据来源与许可见 [`数据字典`](credit_data_dictionary.md)。

19 个原始金融字段：额度 1 个、还款状态 6 个、账单金额 6 个、支付金额 6 个。
派生 5 个，共 24 个模型输入；ID、性别、教育、婚姻、年龄均不入模。

| 派生字段 | 精确口径 | 解释边界 |
|---|---|---|
| utilization_latest | BILL_AMT1 / LIMIT_BAL | 不是收入负债比 |
| utilization_mean | 六期账单均值 / LIMIT_BAL | 负账单保留 |
| delayed_months | 六期状态 >=1 的期数 | 不把 -2/0 自动当缺失 |
| max_delay | 六期状态先截到非负，再取最大值 | 状态码的粗略数值表征 |
| payment_bill_ratio | 六期支付合计 / (六期账单逐项截到非负后的合计 +1) | 不是合同还款率 |

质量表记录无缺失、817 条重复金融历史（相对第一次出现的额外行）、85 个标签冲突组、
3,932 个负账单单元格、2,115 条最新账单超过额度的记录。保留这些记录，不按测试表现裁剪。
同样输入可能有不同标签，不能按结果删去冲突组。全体质量统计只用于描述；
标签关系及相关性图只使用训练集。参见 [`data_quality.json`](../reports/credit/data_quality.json)。

## 3. 模型为什么这样比较

[`models.py`](../src/credit/models.py) 实现 6 个基模型，各保留 raw/sigmoid 两种版本：
原始 LR、WOE LR、balanced LR、HistGBDT、Random Forest、XGBoost；另有训练先验常数基线。
固定容量配置见 [`credit.json`](../configs/credit.json)，没有大规模超参数搜索。

LR 提供线性概率基线；WOE LR 提供可加分箱与评分卡；树模型检验非线性和交互的价值。
balanced LR 是类别权重消融，不能因样本不平衡就默认更优。RF 使用 balanced_subsample，
其 raw PD 偏高；独立校准后概率质量改善。所有候选共享相同的 24 个输入与分区。
没有使用 SMOTE、单调分箱、IV 筛选、拒绝推断补标签或神经网络。

WOE 采用 log(bad share / good share)，不是所有资料都采用的相反符号；EPS=1e-6。
分箱目标 5 箱，稀疏相邻箱合并至满足约束或只剩两箱。所谓零值感知边界不保证
对含负值字段形成纯零箱；没有强制单调性。标准化 LR 系数先还原到 WOE 单位再转分。
600 分对应 good:bad=20:1，PDO=20；积分重构的是 raw WOE LR，而非最终校准 RF。

## 4. 验证设计与防泄漏证据

| 分区 | 行数 / 违约数 | 职责 |
|---|---|---|
| train | 17,990 / 3,977 | 拟合 WOE、填充值、标准化与基模型 |
| calibration | 3,013 / 670 | 对裁剪后的基模型 logit 拟合独立 sigmoid，不重训基模型 |
| validation | 2,999 / 663 | 最小 log loss 选模型、搜索成本阈值、确定通过率阈值 |
| test | 5,998 / 1,326 | 评估冻结的选择与全部候选，不能反过来重新挑选 |

按 19 个金融输入完全一致的历史分组，10 折 StratifiedGroupKFold；0–5/6/7/8–9
分别组成上述四部分。它不是按月份切分，也不是十折训练交叉验证。
相同历史不跨分区降低重复输入污染，但不能证明识别了所有真实关联客户。
行内派生不拟合全样本统计；WOE 是监督变换，必须只在训练集拟合。
SQL 复算 5 个共享派生特征，在 30,000 行上校验与 Python 一致。

验证集同时承担模型和策略选择，存在选择偏差；测试分区已在早期版本公开并复用，
因此只能称为“固定已发布分区的复现结果”，不能声称此次获得全新盲测证据。

## 5. 结果能支持什么

下表取自 [`test_metrics.csv`](../reports/credit/test_metrics.csv)，四舍五入到 4 位：

| 模型 | AUC | KS | AP | Brier | log loss |
|---|---:|---:|---:|---:|---:|
| constant_prior | .5000 | .0000 | .2211 | .1722 | .5283 |
| linear_lr | .7662 | .4366 | .5185 | .1383 | .4402 |
| woe_lr | .7774 | .4308 | .5478 | .1343 | .4298 |
| random_forest | .7917 | .4580 | .5700 | .1724 | .5251 |
| random_forest_sigmoid（选定） | .7917 | .4580 | .5700 | .1322 | .4216 |
| xgboost | .7936 | .4553 | .5728 | .1315 | .4201 |

RF sigmoid 验证 log loss=.413082，低于 XGBoost=.413360；两者很接近，
没有证明 RF 稳健优于所有模型。测试 XGBoost 更优不能用于事后换冠军。
WOE、raw LR 和 HistGBDT 的 sigmoid 版本在此测试表中反而略差，校准不是必然增益。

RF 校准后 AUC 未变、Brier 从 .1724 降至 .1322，说明本次改善主要在概率层。
raw RF 与 sigmoid RF 各自按验证最优阈值决策时，测试混淆矩阵完全相同；
不能把对 WOE 的成本差归因于校准本身。

主情景：拒绝好客户成本 1、放过违约客户成本 5。预测为坏即拒绝：
TP=1000、FP=1601、FN=326、TN=3071，共 5998；
成本=(1601+5×326)/5998=.538680，WOE=.560020，绝对差 .021340 单位/客户。
召回率=1000/1326=75.41%，拒绝中的违约占比=38.45%，通过率=56.64%。
该差约为 WOE 成本的 3.81%，仅是归一化情景比较，不能写作真实坏账或利润提升。
阈值 .171438 由验证集完整同分块搜索得出；理论规则 1/(1+5)=1/6
需概率可信、成本设定正确等假设。见 [`model_policies.csv`](../reports/credit/model_policies.csv)。

500 次配对组 bootstrap 的 95% 百分位区间（[`manifest.json`](../reports/credit/manifest.json)）：
AUC [.7763,.8058]、相对 WOE 的 AUC 差 [.00835,.02036]、模拟成本差 [.00214,.04051]。
它们条件于固定模型与策略，仅反映此测试样本重采样；未包括训练、模型选择和跨期不确定性。

解释对象也须对应：TreeSHAP 是 512 条验证记录的 raw XGBoost log-odds，
不是最终校准 RF 的解释。RF 的验证 permutation importance 仅为描述，相关特征会影响解释。
PSI 的模拟是降额度 30%、非负最新状态加一后重算派生特征；没有新标签，不能推断未来准确率。

## 6. 可复现实验与验收

推荐 Python 3.12（已发布 3.12.14），依赖按版本固定。仓库根目录执行：

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-credit-repro.txt
python -m pytest -q
python scripts/audit.py --negative-control
python scripts/compare_reference.py --run-dir reports/generated/audit/reproduction --output reports/generated/audit/reference_comparison.json
```

`audit.py` 下载并核验归档，执行实验，检查分组覆盖、验证选择和阈值，
从行级预测重算测试指标及混淆成本，并独立打乱四部分标签进行负对照。
单次负对照接近随机不是无泄漏的充分证明。
`compare_reference.py` 再核对已发布和新实验的验证/测试指标、策略表、分区数、
选定模型与主阈值（绝对容差 1e-7），并核对数据、分区 hash 和输入清单。
容差针对已发布原生 float32 challenger 汇总的历史精度差异；每个非零差异均写入 JSON，
不会只以通过掩盖偏差。整数计数变化仍会失败。对账失败时保留差异，不覆盖发布结果。
表格复现一致不代表产生了外部有效性的新证据。

只有退出码成功且检查输出 passed 才算完成各自检查。原始文件在 data/raw；
独立实验中的行数据和模型位于 reports/generated/audit/reproduction/local，保持 Git 忽略。
需要验算命令时可使用 `--run-dir`，但它不会重新训练。
`scripts/run_all.py` 是维护入口，会刷新已发布报告和 README，本流程不调用它。

## 7. 尚未解决、最值得补的研究证据

当前适合作为公开数据上的可复现研究项目。不能称为上线银行风控系统。
优先取得带观测截止日期、字段可用时点及成熟标签的新时期独立队列，再冻结方案做 OOT；
此数据集没有可构造真实 OOT 的多队列日期。缺实际 EAD/LGD、拒绝样本、回收记录、
公平性分组评估、前瞻决策及生产监控，因此无真实金额收益、无公平性或合规认证。
下一步可做特征组消融、状态码类别处理和单调分箱比较，但必须标为新修订，
避免继续在公开测试表上调参后仍把它称作未见测试集。

## 证据导航

| 主张 | 实现 | 可检查结果 |
|---|---|---|
| 来源、24 输入、分区 | src/credit/data.py | split_summary.csv、data_quality.json |
| 固定模型与独立校准 | src/credit/models.py、configs/credit.json | validation_metrics.csv、selection.json |
| 排序/概率与情景决策 | src/credit/evaluation.py、experiment.py | test_metrics.csv、model_policies.csv、cost_sensitivity.csv |
| 条件不确定性 | evaluation.py:bootstrap | manifest.json:intervals |
| 评分卡和模拟 PSI | src/credit/diagnostics.py | scorecard.json、scorecard_bins.csv、stability.json |
| SQL 与 Python 一致 | src/credit/sql_features.py、sql/behavior_features.sql | sql_parity.csv |
| 行级指标重构/负对照 | scripts/audit.py | reports/audit/2026-10-21；新输出 reports/generated/audit |
| 发布表与新实验对账 | scripts/compare_reference.py | 指定的 reference_comparison.json |

上表未带目录的结果均在 reports/credit。旧审计与本次新执行记录应分别引用。
