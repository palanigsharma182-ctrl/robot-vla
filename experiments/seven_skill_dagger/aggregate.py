"""合并互斥train采集分片；保留来源身份与失败分母，不复制生成新监督。"""
import json
from pathlib import Path
import shutil
from experiments.tcp_atomic_skills.protocol import save, sha


def aggregate(roots, output, skill, expected_student_sha):
    output = Path(output)
    if output.exists():
        raise ValueError('合并必须使用新目录')
    seen = set()
    records = []
    sources = []
    copies = []
    for part, root in enumerate(map(Path, roots)):
        result = json.loads((root/'collection.json').read_text())
        if (result['status'] != 'completed' or result['mode'] != 'collect'
                or result['checkpoint_sha256'] != expected_student_sha):
            raise ValueError('分片未完成或学生身份不一致')
        sources.append(dict(collection_sha256=sha(root/'collection.json'),source_sha256=result['source_sha256']))
        for original in result['records']:
            row = dict(original)
            unit = (row['seed'], row['skill'], row['case'])
            if row['split'] != 'train' or row['skill'] != skill or unit in seen:
                raise ValueError('重复采集单元或跨技能/开发数据')
            seen.add(unit)
            for key in ('trajectory','labels','boundaries'):
                if key not in row:
                    continue
                source = root/row[key]
                if not source.resolve().is_relative_to(root.resolve()) or sha(source) != row[key+'_sha256']:
                    raise ValueError('分片路径或SHA错误')
                target = Path(f'part-{part}')/row[key]
                copies.append((source, output/target))
                row[key] = str(target)
            records.append(row)
    output.mkdir(parents=True)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    result = dict(schema='independent-skill-correction-aggregate/v1', status='completed', mode='collect',
                  checkpoint_sha256=expected_student_sha, sources=sources, records=records)
    save(output/'collection.json',result)
    return result
