"""独立关键点定位器训练；RGB 是唯一模型输入，GT 仅作为监督标签。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.align_precision_vision.vision import AlignLocalizer,localization_loss


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class KeypointTrainingData:
    """按场景隔离 train/development；development 文件不在训练中打开。"""
    def __init__(self,manifest):
        self.path=Path(manifest);self.identity=digest(self.path)
        data=json.loads(self.path.read_text())
        if (data.get('schema')!='align-precision-keypoints/v1'
            or data.get('object_model')!='upright-cube-4cm'
            or data.get('pixel_convention')!='zero-based-pixel-centers'):
            raise ValueError('只接受明确物体与像素语义的关键点数据')
        ids=set();paths=set();scenes={};self.rows=[]
        for row in data['records']:
            if row['split'] not in ('train','development'):raise ValueError('不消费 final test')
            if row['id'] in ids or row['file'] in paths:raise ValueError('重复样本或文件')
            ids.add(row['id']);paths.add(row['file'])
            if row['scene'] in scenes and scenes[row['scene']]!=row['split']:raise ValueError('场景跨训练/开发集合')
            scenes[row['scene']]=row['split']
            relative=Path(row['file']);file=(self.path.parent/relative).resolve()
            if relative.is_absolute() or not file.is_relative_to(self.path.parent.resolve()):raise ValueError('样本路径越界')
            if row['split']=='train':
                if digest(file)!=row['sha256']:raise ValueError('训练样本哈希不符')
                self.rows.append((row,file))
        if not self.rows:raise ValueError('训练集为空')

    def get(self,index,device):
        row,path=self.rows[index]
        with np.load(path,allow_pickle=False) as data:
            rgb=data['rgb'];uv=data['pixel_uv'];visible=data['visible']
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('RGB格式错误')
        if uv.shape!=(8,2) or visible.shape!=(8,) or visible.dtype!=np.bool_:raise ValueError('八关键点标签格式错误')
        image=torch.tensor(rgb.transpose(2,0,1).copy(),device=device,dtype=torch.float32)[None]/255
        return image,torch.tensor(uv,device=device,dtype=torch.float32)[None],torch.tensor(visible,device=device)[None]


def train(manifest,output,*,steps,wall_seconds,device='cuda',seed=42,learning_rate=1e-3,channels=(32,64,128,256)):
    if steps<=0 or wall_seconds<=0 or not np.isfinite([wall_seconds,learning_rate]).all() or learning_rate<=0:
        raise ValueError('步数、时间和学习率必须有限且为正')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();data=KeypointTrainingData(manifest)
    torch.manual_seed(seed);model=AlignLocalizer(channels=channels).to(device);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    config=dict(format='align-precision-localizer/v1',manifest_sha256=data.identity,steps=steps,
        seed=seed,learning_rate=learning_rate,channels=list(channels),device=device,
        selection='fixed final update; train only',wall_seconds=wall_seconds,
        source_sha256={name:digest(Path(__file__).parent/name) for name in ('train.py','vision.py','geometry.py')})
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rng=np.random.default_rng(seed);schedule=[]
    while len(schedule)<steps:schedule.extend(rng.permutation(len(data.rows)).tolist())
    schedule=schedule[:steps];(output/'schedule.json').write_text(json.dumps(schedule)+'\n')
    losses=[]
    try:
        for step,index in enumerate(schedule,1):
            if time.monotonic()-started>wall_seconds:raise TimeoutError('定位器训练达到预定时间上限')
            image,uv,visible=data.get(index,device);optimizer.zero_grad(set_to_none=True)
            loss=localization_loss(model(image),uv,visible)
            if not torch.isfinite(loss):raise ValueError('非有限定位损失')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        payload=dict(format=config['format'],model=model.state_dict(),optimizer=optimizer.state_dict(),
            completed=True,step=steps,config=config,losses=losses,torch_rng=torch.get_rng_state())
        path=output/'final.pt';torch.save(payload,path)
        restored=torch.load(path,map_location=device,weights_only=True)
        model.load_state_dict(restored['model'],strict=True)
        result=dict(status='completed',steps=steps,strict_reload=True,checkpoint_sha256=digest(path),
                    elapsed_s=time.monotonic()-started,accepted=False)
    except BaseException as exc:
        result=dict(status='error',completed_updates=len(losses),error=f'{type(exc).__name__}: {exc}')
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');raise
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--wall-seconds',type=int,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();print(json.dumps(train(**vars(args))))


if __name__=='__main__':main()
