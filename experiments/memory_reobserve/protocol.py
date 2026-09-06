"""本批实验参数与采集身份；独立于可执行入口，避免双模块类身份。"""
from dataclasses import dataclass

SEEDS = tuple(range(1010100, 1010132))
PROTOCOL = dict(id='memory-reobserve-five-skills/development-v2', seeds=list(SEEDS),
    train_seeds=list(SEEDS[:24]), development_seeds=list(SEEDS[24:]),
    maximum_collection_seconds=600, return_steps=10, maximum_teacher_steps=64,
    anchor_stride=4, horizon=16, memory_max_age_s=2.5,
    initial_teacher_hold_steps={str(seed):28 if seed%2==0 else 0 for seed in SEEDS},
    timing_revision='v1 had no available-to-stale action anchors; v2 adds actual HOME holds, preserves v1 evidence',
    acquisition='fixed single PRIMARY request; qualified D049 commit or masked HOME sample',
    threshold_changes=False, replacement_seeds=False,
    train_steps_per_arm=1024, learning_rate=1e-5, accumulation_steps=4,
    replay_fraction=.75, memory_dropout=.25, sampling_seed=42,
    frozen='Qwen + Adapter; common restored layer12 Expert initialization',
    selection='last step; all seeds and failures retained; development only, no final test')


@dataclass(frozen=True)
class Acquisition:
    seed: int

    def __post_init__(self):
        if self.seed not in SEEDS:
            raise ValueError('只允许本批冻结的新数据场景')

    def capture(self, **kwargs):
        from experiments.memory_reobserve.collect import capture_sequence
        return capture_sequence(seed=self.seed, **kwargs)
