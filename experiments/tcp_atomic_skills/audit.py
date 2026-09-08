"""训练前检查完整分母、五技能采样和物理标签；不加载模型。"""
import argparse
from pathlib import Path
from experiments.tcp_atomic_skills.data import load_examples
from experiments.tcp_atomic_skills.protocol import save, verify_source
from experiments.tcp_memory_control.kinematics import TCPKinematics


def main():
    p = argparse.ArgumentParser()
    for key in ('collection', 'output', 'source-manifest'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(); source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False)
    rows, provenance = load_examples(args.collection, TCPKinematics())
    save(args.output/'result.json', dict(status='passed', source_sha256=source, data=provenance,
        samples={s:len(v) for s,v in rows.items()},
        skill_samples={s:[sum(x['skill_id']==i for x in v) for i in range(5)] for s,v in rows.items()}))
    print('Dataset and TCP label audit passed', flush=True)


if __name__ == '__main__':
    main()
