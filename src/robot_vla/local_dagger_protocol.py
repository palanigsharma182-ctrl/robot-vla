"""Local DAgger action-budget protocol 的冻结定义与纯决策逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD = "action_budget_protocol"
LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD = "action_budget_usage"
LOCAL_DAGGER_ACTION_BUDGET_CONTRACT_VERSION = (
    "robot-vla-local-dagger-segmented-action-budget/v1"
)
LOCAL_DAGGER_ACTION_BUDGET_DEADLINE_SEMANTICS = (
    "success_must_precede_environment_truncation"
)
LOCAL_DAGGER_ACTION_BUDGET_UNIT = "actual_environment_action"

POLICY_ACTION_BUDGET_EXHAUSTED_REASON = (
    "Policy 达到 300 Action 预算但未到达目标 boundary"
)
EXPERT_ACTION_BUDGET_EXHAUSTED_REASON = (
    "Expert takeover 后达到 180 Action 恢复预算但任务未成功"
)


class LocalDaggerActionBudgetProtocol(str, Enum):
    """E012 Local DAgger 可选的固定 action-budget protocol。"""

    LEGACY = "legacy"
    SEGMENTED_300_180_480 = "segmented-300-180-480"


@dataclass(frozen=True)
class LocalDaggerActionBudgetPlan:
    """Protocol 的冻结上限；``None`` 表示继续使用环境的 legacy TimeLimit。"""

    protocol: LocalDaggerActionBudgetProtocol
    policy_action_limit: int | None
    expert_action_limit: int | None
    environment_action_limit: int | None

    @property
    def amended(self) -> bool:
        return self.protocol is not LocalDaggerActionBudgetProtocol.LEGACY

    def planned_metadata(self) -> dict[str, Any] | None:
        """仅 amended protocol 写 provenance；legacy 必须完全省略。"""

        if not self.amended:
            return None
        return {
            "version": LOCAL_DAGGER_ACTION_BUDGET_CONTRACT_VERSION,
            "name": self.protocol.value,
            "action_unit": LOCAL_DAGGER_ACTION_BUDGET_UNIT,
            "policy_action_limit": self.policy_action_limit,
            "expert_action_limit": self.expert_action_limit,
            "environment_action_limit": self.environment_action_limit,
            "deadline_semantics": (
                LOCAL_DAGGER_ACTION_BUDGET_DEADLINE_SEMANTICS
            ),
        }

    def usage_metadata(
        self,
        *,
        total_actions: int,
        expert_takeover_step: int | None,
    ) -> dict[str, int] | None:
        """按真实已执行 Action 构造互斥 Policy/Expert usage。"""

        if not self.amended:
            return None
        if total_actions < 0:
            raise ValueError("Local DAgger total_actions 不能为负数")
        if expert_takeover_step is None:
            policy_actions = total_actions
            expert_actions = 0
        else:
            if not 0 <= expert_takeover_step <= total_actions:
                raise ValueError("Expert takeover step 超出已执行 Action")
            policy_actions = expert_takeover_step
            expert_actions = total_actions - expert_takeover_step
        return {
            "policy_actions": policy_actions,
            "expert_actions": expert_actions,
            "total_actions": total_actions,
        }

    def policy_budget_exhausted_after_action(
        self,
        *,
        policy_actions: int,
        boundary_reached: bool,
    ) -> bool:
        """Boundary crossing 优先于恰好同一步耗尽 Policy budget。"""

        if policy_actions < 0:
            raise ValueError("policy_actions 不能为负数")
        return bool(
            self.policy_action_limit is not None
            and policy_actions >= self.policy_action_limit
            and not boundary_reached
        )

    def expert_budget_exhausted_after_action(
        self,
        *,
        expert_actions: int,
        task_completed: bool,
        truncated: bool,
    ) -> bool:
        """环境 truncation 保持原分类；非 truncated 的 success 优先于 budget。"""

        if expert_actions < 0:
            raise ValueError("expert_actions 不能为负数")
        return bool(
            self.expert_action_limit is not None
            and expert_actions >= self.expert_action_limit
            and not truncated
            and not task_completed
        )

    def environment_hard_deadline_reached_after_action(
        self,
        *,
        total_actions: int,
    ) -> bool:
        """第 environment_action_limit 个 Action 已被 TimeLimit 截断。"""

        if total_actions < 0:
            raise ValueError("total_actions 不能为负数")
        return bool(
            self.environment_action_limit is not None
            and total_actions >= self.environment_action_limit
        )


def resolve_local_dagger_action_budget(
    protocol: LocalDaggerActionBudgetProtocol | str,
) -> LocalDaggerActionBudgetPlan:
    """将 CLI/API 值解析成唯一冻结计划。"""

    try:
        resolved = LocalDaggerActionBudgetProtocol(protocol)
    except ValueError as exc:
        choices = ", ".join(item.value for item in LocalDaggerActionBudgetProtocol)
        raise ValueError(f"未知 Local DAgger action-budget protocol；可选: {choices}") from exc
    if resolved is LocalDaggerActionBudgetProtocol.LEGACY:
        return LocalDaggerActionBudgetPlan(
            protocol=resolved,
            policy_action_limit=None,
            expert_action_limit=None,
            environment_action_limit=None,
        )
    return LocalDaggerActionBudgetPlan(
        protocol=resolved,
        policy_action_limit=300,
        expert_action_limit=180,
        environment_action_limit=480,
    )


__all__ = [
    "EXPERT_ACTION_BUDGET_EXHAUSTED_REASON",
    "LOCAL_DAGGER_ACTION_BUDGET_CONTRACT_VERSION",
    "LOCAL_DAGGER_ACTION_BUDGET_DEADLINE_SEMANTICS",
    "LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD",
    "LOCAL_DAGGER_ACTION_BUDGET_UNIT",
    "LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD",
    "POLICY_ACTION_BUDGET_EXHAUSTED_REASON",
    "LocalDaggerActionBudgetPlan",
    "LocalDaggerActionBudgetProtocol",
    "resolve_local_dagger_action_budget",
]
