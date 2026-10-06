# 简历项目描述 / Resume descriptions

仅保留自己能讲清楚、亲自复现和检查过的表述。数字由本次运行结果自动填入，
不把公开数据实验称为银行实际业务、上线系统或实际利润。

## 中文（完整）

**信用卡违约风险建模与评分卡研究｜Python、SQL、scikit-learn、XGBoost**

- 基于 UCI 30,000 条历史信用卡客户记录，完成数据质量检查、六期行为特征构建和分组训练/校准/验证/测试流程，避免重复金融记录跨集合。
- 实现 WOE/IV、Logistic 评分卡及逐箱分数映射，与 Random Forest、XGBoost 等模型比较，结合 ROC-AUC、KS、PR 指标、Brier 和可靠性曲线评估排序与概率质量；验证集选定模型的测试 AUC 为 {{AUC}}、KS 为 {{KS}}。
- 在明确的非对称成本假设下选择决策阈值，完成特征/分数 PSI 与模拟漂移诊断；执行 SQL 行为聚合并验证五项特征与 Python 在全部样本上一致，导出可复现报告和批量推理模型。

## 中文（精简，一段）

基于 UCI 30,000 条历史信用卡记录构建信用违约建模流程，实现 WOE/IV 与 Logistic 评分卡，比较 Random Forest、XGBoost 等模型，并评估概率校准、成本阈值和 PSI。采用分组的训练/校准/验证/测试设计，验证集选定模型测试 AUC {{AUC}}、KS {{KS}}；完成 SQL/Python 特征一致性检查及可复现实验报告。

## English

**Credit Default Modeling and Scorecard Research | Python, SQL, scikit-learn, XGBoost**

- Built a reproducible credit-default study on 30,000 UCI cardholder records, with quality audits, six-period behavioral features and grouped training/calibration/validation/test partitions.
- Implemented a WOE/IV logistic scorecard and compared Random Forest, XGBoost and other baselines; the validation-selected model achieved test ROC-AUC {{AUC}} and KS {{KS}}, with probability quality assessed using Brier score and reliability curves.
- Evaluated asymmetric-cost thresholds and feature/score PSI under explicit scenarios; checked five SQL/Python features across all records and exported research reports and local batch-scoring artifacts.

## 数字口径和面试核对

- 主模型：`{{SELECTED}}`，依据验证集 Log loss 选定，未根据测试表重选。
- 测试 AP：{{AP}}；Brier：{{BRIER}}。测试分数沿用此前公开的划分，不称作全新独立验证。
- 误拒好客户成本 1、放行坏客户成本 5 的情景：阈值 {{THRESHOLD}}；通过率 {{APPROVAL}}；坏样本召回率 {{RECALL}}；每客户归一化成本 {{COST}}，WOE 基准 {{WOE_COST}}。
- “成本”是设定假设下的离线核算，不写“为银行节省X万元”或“提升实际收入X%”。
- 没有真实 OOT、拒贷标签、实际损失金额或生产部署，不在简历中声称实现。
