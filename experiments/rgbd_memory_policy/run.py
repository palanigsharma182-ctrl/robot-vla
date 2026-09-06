"""隔离阶段入口；累计资源上限由外层私有runner执行，阶段各用独立输出目录。"""
import argparse
from pathlib import Path
import signal


def main():
    def stopped(_signal,_frame):
        raise TimeoutError('外层累计预算停止，保留本阶段已执行记录')
    signal.signal(signal.SIGTERM,stopped)
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['collect','train','evaluate'])
    for name in ('output','data','checkpoint','model-cache','training','source-manifest'):
        p.add_argument('--'+name,type=Path,required=name=='output')
    a=p.parse_args()
    if a.stage=='collect':
        from experiments.rgbd_memory_policy.collect import collect
        collect(a.output)
    elif a.stage=='train':
        from experiments.rgbd_memory_policy.train import train
        train(a.data,a.checkpoint,a.model_cache,a.output,a.source_manifest)
    else:
        from experiments.rgbd_memory_policy.evaluate import evaluate
        evaluate(a.checkpoint,a.model_cache,a.training,a.output,a.source_manifest)


if __name__=='__main__':main()
