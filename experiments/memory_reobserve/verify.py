"""四个新场景的一次观察工程验证，失败不换 seed，不冒充任务效果。"""
import argparse
import json
from pathlib import Path
import time

from experiments.g2c_memory_integration.run import run
from experiments.memory_reobserve.runtime import load_memory_runtime
from experiments.memory_reobserve.session import ENGINEERING_SEEDS, MemoryRouteSession


def main():
    p = argparse.ArgumentParser()
    for name in ('bundle', 'upstream', 'model-cache', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime, identity = load_memory_runtime(a.upstream, a.model_cache, a.candidate, a.candidate_sha256)
    (a.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    records = []
    for seed in ENGINEERING_SEEDS:
        if time.monotonic()-started > 540:
            records.append(dict(seed=seed, status='budget-not-run'))
            continue
        try:
            session = MemoryRouteSession(seed, runtime)
            result = run(a.bundle, a.output/str(seed), memory_session=session)
            audit=json.loads((a.output/str(seed)/'memory-runtime.json').read_text())
            checks=dict(one_request=sum(r['decision']['requestable'] for r in audit['trigger'])==1,
                one_commit=result['memory_write_count']==1,
                available_memory_read=any(r['status']=='consumed' and r['snapshot']['available'] for r in audit['reads']),
                memory_action_sent=any(r['available'] for r in audit['sends']),
                resumed=result['vla_resumed'],
                expired_history_cleared=bool(audit['cleanup']) and all(r['all_history_empty'] for r in audit['cleanup']))
            status='qualified-passed' if all(checks.values()) else 'no-commit' if not checks['one_commit'] else 'incomplete-chain'
            records.append(dict(seed=seed, status=status, checks=checks, result=result))
        except Exception as e:
            records.append(dict(seed=seed, status='error', error_type=type(e).__name__, error=str(e)))
        (a.output/'result.json').write_text(json.dumps(dict(records=records,
            elapsed_s=time.monotonic()-started, scope='capability-absent engineering only',
            new_training_steps=0, protected_test=False), indent=2)+'\n')
    qualified=sum(r['status']=='qualified-passed' for r in records)
    print(json.dumps(dict(planned=len(ENGINEERING_SEEDS), qualified=qualified,
        errors=sum(r['status']=='error' for r in records), elapsed_s=time.monotonic()-started)))
    if qualified != len(ENGINEERING_SEEDS):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
