"""真实接触环境采集完整教师任务，保留每个预定场景与命令来源。"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

from experiments.tcp_atomic_skills.protocol import TRAIN_SEEDS, DEV_SEEDS, PROTOCOL, save, sha, verify_source


def main():
    from robot_vla.sim.collector import TrustedPickPlaceCollector, EpisodeRejected
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    args = p.parse_args()
    source = verify_source(args.source_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    records = [dict(seed=s, split=split, status='not_run')
               for split, seeds in [('train', TRAIN_SEEDS), ('val', DEV_SEEDS)] for s in seeds]
    start = time.monotonic()
    result = dict(status='running', protocol=PROTOCOL, source_sha256=source, records=records)
    def persist():
        result['elapsed_s'] = time.monotonic() - start
        save(args.output/'collection.json', result)
    persist()
    try:
        with TrustedPickPlaceCollector(args.output/'dataset', max_episode_steps=600) as collector:
            for row in records:
                row['status'] = 'running'; persist()
                try:
                    meta = collector.collect(seed=row['seed'], split=row['split'])
                    row.update(status='completed', trajectory_id=meta.trajectory_id,
                               file=meta.file, sha256=sha(args.output/'dataset'/meta.file),
                               steps=meta.num_steps, outcome=meta.outcome_evidence.to_dict())
                except EpisodeRejected as error:
                    row.update(status='teacher_rejected', error=str(error))
                persist()
                print(__import__('json').dumps(row, ensure_ascii=False), flush=True)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
