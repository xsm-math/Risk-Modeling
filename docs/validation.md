# 发布前验证

验证日期：2026-09-30。

## 环境

Windows、Python 3.12.14、NumPy 2.5.3、pandas 3.0.1、pytest 9.1.1、scikit-learn 1.9.1。
主要依赖精确版本见 `requirements-repro.txt`；常规运行与开发依赖分别见 `requirements.txt`、`requirements-dev.txt`。

## 已完成

- `python -m pytest -q`：22 passed，含两项 sklearn 对照测试，无跳过。
- 两次连续执行 `src.train.run()`，完整返回的汇总字典一致。
- 核对 AUC 梯形积分和平均秩结果一致。
- 核对测试成本 = 243 × 400 + 35 × 8,200 = 384,200。
- 评分卡原始系数能重构模型概率；逐客输出含有限的分数。
- PSI 在同分布时为零，常量/稀疏特征漂移能触发告警。
- 端到端回归测试验证 WOE 只在 4,481 条拟合集记录上调用 fit。
- 仓库不包含原始客户数据、逐客预测或环境凭证。

GitHub Actions 配置了 Python 3.11 / 3.12 的测试与训练任务。云端结果以仓库 Actions 页面为准，本记录只证明本地验证。
