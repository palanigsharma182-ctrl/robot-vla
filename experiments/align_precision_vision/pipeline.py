"""同帧精确视觉到 Align 条件；Action Expert 是唯一学习动作输出端。"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from experiments.align_precision_vision.geometry import (
    GeometrySpec, PoseMeasurement, measure_keypoints, align_condition, yaw_symmetries,
)


@dataclass(frozen=True)
class PrecisionFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    base_from_camera_cv: np.ndarray
    episode: str
    target: str
    timestamp_s: float
    depth_timestamp_s: float
    calibration_timestamp_s: float
    camera_convention: str = 'opencv-optical-x-right-y-down-z-forward'

    def __post_init__(self):
        if self.rgb.ndim!=3 or self.rgb.shape[2]!=3 or self.rgb.dtype!=np.uint8:
            raise ValueError('必须提供原始 uint8[H,W,3] RGB')
        if self.depth_m.shape!=self.rgb.shape[:2] or not np.issubdtype(self.depth_m.dtype,np.floating):
            raise ValueError('深度必须与RGB同分辨率，以浮点米表示')
        if self.camera_convention!='opencv-optical-x-right-y-down-z-forward':
            raise ValueError('相机必须先显式转为 OpenCV optical frame')
        if not self.episode or not self.target or not all(math.isfinite(t) for t in (
            self.timestamp_s,self.depth_timestamp_s,self.calibration_timestamp_s)):
            raise ValueError('身份/时间无效')


class AlignPrecisionPipeline:
    """仅适用单个已知4cm方块；target身份由上层目标关联提供，不在这里猜测。"""
    def __init__(self,localizer,object_from_pregrasp,*,spec=GeometrySpec()):
        from experiments.tcp_memory_control.geometry import pose
        self.localizer=localizer;self.grasp=pose(object_from_pregrasp);self.spec=spec
        if not np.allclose(self.grasp[:2,3],0,atol=1e-7) or not np.isclose(abs(self.grasp[2,2]),1,atol=1e-7):
            raise ValueError('第一版四重对称只支持沿物体Z轴的中央抓取前目标')

    @torch.no_grad()
    def condition(self,frame,base_from_tcp,*,episode,target,now_s,tcp_timestamp_s):
        times=(frame.timestamp_s,frame.depth_timestamp_s,frame.calibration_timestamp_s)
        current_time=float(now_s() if callable(now_s) else now_s)
        if not math.isfinite(current_time):raise ValueError('当前时间必须有限')
        if max(times)>current_time+1e-9:
            measurement=PoseMeasurement(frame.episode,frame.target,frame.timestamp_s,None,'future_observation')
        elif max(times)-min(times)>self.spec.max_sensor_skew_s:
            measurement=PoseMeasurement(frame.episode,frame.target,frame.timestamp_s,None,'rgbd_calibration_skew')
        else:
            self.localizer.eval()
            device=next(self.localizer.parameters()).device
            rgb=torch.from_numpy(np.ascontiguousarray(frame.rgb.transpose(2,0,1))).to(device=device,dtype=torch.float32)[None]/255
            output=self.localizer(rgb);uv,visibility=output.decode()
            measurement=measure_keypoints(uv[0].cpu().numpy(),visibility[0].cpu().numpy(),frame.depth_m,
                frame.intrinsic,frame.base_from_camera_cv,episode=frame.episode,target=frame.target,
                timestamp_s=frame.timestamp_s,spec=self.spec)
        condition=align_condition(measurement,base_from_tcp,self.grasp,episode=episode,target=target,
            now_s=float(now_s() if callable(now_s) else now_s),tcp_timestamp_s=tcp_timestamp_s,spec=self.spec,symmetries=yaw_symmetries())
        return measurement,condition

    @torch.no_grad()
    def predict(self,expert,context,proprio,noise,action_mask,frame,base_from_tcp,*,episode,target,now_s,tcp_timestamp_s):
        """同一规划点生成条件与 chunk；不发送动作，不改变原IK与执行器。"""
        if not callable(now_s):raise ValueError('在线预测必须提供同一传感器时基的时钟函数')
        if proprio.shape!=(1,15):raise ValueError('在线 Align 接口固定 batch=1、proprio=15')
        measurement,condition=self.condition(frame,base_from_tcp,episode=episode,target=target,
            now_s=now_s,tcp_timestamp_s=tcp_timestamp_s)
        expert.eval()
        action=expert.predict(context,proprio,noise,action_mask,conditions=[condition],mode='measured')
        finished=float(now_s())
        if not math.isfinite(finished) or finished-min(frame.timestamp_s,tcp_timestamp_s)>self.spec.max_age_s:
            raise TimeoutError('推理期间观测已过期，本次chunk不可执行')
        return {'action':action,'measurement':measurement,'condition':condition}
