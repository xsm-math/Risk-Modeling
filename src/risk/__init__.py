"""信用风险建模工具包。

模块导航：
- config   业务假设与超参数（改这里就能改实验）
- data     数据生成/读取
- features 分箱 / WOE / IV / 评分卡刻度
- model    纯 numpy 逻辑回归
- metrics  AUC / KS / PSI / 期望损失 / 最优阈值
- monitor  特征与分数 PSI 监控
"""
__all__ = ["config", "data", "features", "model", "metrics", "monitor"]
__version__ = "0.1.0"
