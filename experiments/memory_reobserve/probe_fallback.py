"""已见场景的失效后视觉回退工程验证，不作为新的任务收益证据。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from experiments.g2c_memory_integration.run import run
from experiments.memory_reobserve.runtime import (
    load_memory_runtime, load_trained_runtime, MemoryConditionedRuntime, M0_CANDIDATE_SHA256,
)
from experiments.memory_reobserve.session import MemoryRouteSession


def main():
    parser = argparse.ArgumentParser()
    for name in ('bundle', 'upstream', 'model-cache', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--training-result', type=Path,
        help='本批训练 result.json；省略时仍严格使用旧 M0 权重')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(seed=1001202, return_steps=10, post_replans=12, starting_sample_index=2,
        max_memory_age_s=2.5, visual_fallback=True, maximum_process_seconds=180,
        candidate_kind='trained-memory' if args.training_result else 'previous-m0',
        scope='post-hoc engineering: fresh masked visual replan after memory invalidation; no task-effect claim')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    if args.training_result:
        trained = json.loads(args.training_result.read_text())
        if trained['status'] != 'training-completed':
            raise ValueError('仅验证本批已完成的训练权重')
        base, identity = load_trained_runtime(args.upstream, args.model_cache, args.candidate,
            trained['arms']['memory']['checkpoint_sha256'],
            expected_training_identity_sha256=trained['training_identity_sha256'],
            expected_arm='memory')
    else:
        base, identity = load_memory_runtime(args.upstream, args.model_cache, args.candidate, M0_CANDIDATE_SHA256)
    runtime = MemoryConditionedRuntime(base.policy, base.processor_adapter, base.proprio_normalizer,
        base.spec, base.device, replace(base.config, starting_sample_index=2))
    session = MemoryRouteSession(protocol['seed'], runtime, return_steps=10, post_replans=12, visual_fallback=True)
    result = run(args.bundle, args.output/'episode', memory_session=session)
    reads = runtime.memory_reads
    available = [i for i,r in enumerate(reads) if r['snapshot']['available']]
    fallback = [r for r in reads[max(available)+1:]] if available else []
    checks = dict(memory_used=bool(available), fallback_has_fresh_masked_inference=bool(fallback)
        and all(not r['snapshot']['available'] and not any(r['snapshot']['features']) for r in fallback),
        fresh_time=all(b['snapshot']['timestamp_s'] > a['snapshot']['timestamp_s'] for a,b in zip(reads,reads[1:])),
        old_actions_cleared=any(r.get('all_history_empty') and r.get('next_mode')=='visual-only' for r in session.cleanup_records),
        visual_actions_after_fallback=any(r['conditioning']=='visual-only' and r['stage']=='continued-execution'
            and r['execution']['executed_steps']>0 for r in result['vla_execution']),
        no_second_observation=result['memory_write_count']==1 and len([r for r in session.trigger_records if r['decision']['requestable']])==1)
    output = dict(status='passed' if all(checks.values()) else 'failed', protocol=protocol, checks=checks,
        elapsed_s=time.monotonic()-started, identity=identity,
        available_inferences=len(available), fallback_inferences=len(fallback),
        execution=result['vla_execution'], cleanup=session.cleanup_records)
    (args.output/'result.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('status','checks','elapsed_s','available_inferences','fallback_inferences')}))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__=='__main__':
    main()
