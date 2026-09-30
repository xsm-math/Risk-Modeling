"""可复现实验配置与显式成本假设。"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- 数据生成 ----
    seed: int = 42
    n_samples: int = 8000
    # 生成数据时的"真实"先生截距（数值模拟的 logistic 截距）
    base_logit: float = -1.75  # 与特征分布共同决定坏样本率
    noise_scale: float = 0.25  # DGP 噪声（越小 AUC 越高）

    # ---- 数据切分 ----
    test_size: float = 0.30       # 训练/测试 7:3
    val_size: float = 0.20        # 训练集内部再切 20% 做验证（只用于选阈值）

    # ---- WOE 分箱：沿用历史基线配置，其选择偏差见 docs/methodology.md ----
    # 本次不再按测试表现调参。
    max_bins: int = 8
    min_bin_pct: float = 0.03     # 每箱至少占样本 3%，否则合并
    min_bin_bad: int = 20         # 每箱至少 20 个坏样本，否则合并（低坏账率数据的稳定器）
    iv_threshold: float = 0.02    # IV 低于此值的特征直接丢弃（区分度太弱）

    # ---- 建模 ----
    l2: float = 0.01              # L2 正则强度（对应高斯先验，抑制大权重）
    lr: float = 0.5
    epochs: int = 400
    tol: float = 1e-7             # 损失变化小于 tol 提前停止

    # ---- 业务假设（用来算最优阈值与 ROI）----
    # 拒绝一个会违约的客户 = 避免损失（假阴性成本，越大越倾向拒）
    loss_per_default: float = 8000.0
    # 拒绝一个好客户 = 损失这笔业务的利润（假阳性成本）
    profit_per_good: float = 400.0
    # 通过一个会违约的客户，除本金损失外的额外催收/运营成本
    extra_cost_per_bad_through: float = 200.0

    # ---- PSI 监控 ----
    psi_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25

    @property
    def total_cost_fp(self) -> float:
        """把一个好客户误拒的总代价。"""
        return self.profit_per_good

    @property
    def total_cost_fn(self) -> float:
        """把一个坏客户误放的总代价。"""
        return self.loss_per_default + self.extra_cost_per_bad_through


CFG = Config()
