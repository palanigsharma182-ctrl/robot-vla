"""新联合候选入口；命令行不隐含资源预算，外层按整个任务累计限制。"""
import argparse
from pathlib import Path
import signal


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['train','evaluate','smoke','compare-baseline'])
    for name in ['output','source-manifest']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['checkpoint','model-cache','data','training','training-source-manifest']:
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    args.compare_baseline=args.stage=='compare-baseline'
    if args.compare_baseline:args.stage='evaluate'
    required={'train':['checkpoint','model_cache','data'],'evaluate':['checkpoint','model_cache','training'],'smoke':[]}[args.stage]
    if any(getattr(args,key) is None for key in required):p.error('缺少该阶段必要输入: '+','.join(required))
    def stop(*_):raise TimeoutError('本轮资源上限触发，保存现有证据并停止')
    signal.signal(signal.SIGTERM,stop)
    if args.stage=='train':
        from experiments.tcp_memory_control.train import train
        train(args)
    else:
        from experiments.tcp_memory_control.evaluate import evaluate,smoke
        (evaluate if args.stage=='evaluate' else smoke)(args)


if __name__=='__main__':main()
