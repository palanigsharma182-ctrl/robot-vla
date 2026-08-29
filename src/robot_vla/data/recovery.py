"""可信失败恢复轨迹的版本化数据契约。"""

RECOVERY_CONTRACT_VERSION = "trusted-pick-place-recovery/v1"
RECOVERY_PROFILES = ("reach", "grasp", "lift", "transport", "place")

__all__ = ["RECOVERY_CONTRACT_VERSION", "RECOVERY_PROFILES"]
