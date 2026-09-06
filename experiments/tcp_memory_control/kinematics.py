"""复用已安装SAPIEN/URDF的显式FK和IK；策略不学习机器人运动学。"""
import numpy as np
from scipy.spatial.transform import Rotation
from robot_vla.contracts import RobotSpec
from robot_vla.diagnostics.oracle_reach import FrankaTCPForwardKinematics, find_maniskill_panda_urdf
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from experiments.tcp_memory_control.geometry import TCPActionSpec, pose, apply_delta


class TCPKinematics(FrankaTCPForwardKinematics):
    def __init__(self):
        super().__init__(find_maniskill_panda_urdf(), RobotSpec())

    def inverse(self, target, reference_q):
        import sapien
        target = pose(target);reference_q = self._validated_arm_q(reference_q)
        q = np.r_[reference_q, 0., 0.]
        answer, success, error = self._model.compute_inverse_kinematics(
            self._tcp_link_index, sapien.Pose(target), initial_qpos=q,
            active_qmask=np.array([1]*7+[0]*2, dtype=np.int32),
            eps=1e-5, max_iterations=100, dt=.5, damp=1e-6)
        answer = np.asarray(answer)[:7]
        bounds = np.asarray(self.spec.joint_position_limits_rad)
        if (not success or not np.isfinite(answer).all()
            or np.any(answer < bounds[:, 0]) or np.any(answer > bounds[:, 1])):
            raise ValueError('IK失败或超出关节范围，不返回伪成功动作')
        actual = self.pose_base(answer)
        if (np.linalg.norm(actual[:3, 3]-target[:3, 3]) > 1e-4
            or np.linalg.norm(Rotation.from_matrix(actual[:3,:3] @ target[:3,:3].T).as_rotvec()) > 1e-3):
            raise ValueError('IK解未通过FK回代')
        return answer.astype(np.float32)


class TCPChunkExecutor:
    """把TCP chunk前缀转成joint目标后交给现有执行器，沿用20Hz与逐步观察。

    每次重规划显式从actual状态重建参考；训练第一步标签使用相同规则。
    IK仅求解即将执行的四步；任一步失败则整个前缀不发送。
    """
    def __init__(self, kinematics):
        self.kinematics = kinematics
        self.tcp_spec = TCPActionSpec()
        self.joint = RecedingHorizonChunkExecutor(RobotSpec())
        self.last_targets = []

    def reset(self):
        self.joint.reset()

    def execute(self, physical, controller, anchor):
        self.tcp_spec.normalize(physical)
        if np.asarray(physical).shape != (16, 7):
            raise ValueError('TCP chunk必须为[16,7]')
        self.joint.reset()
        reference_q = controller.read_state().joint_positions.copy()
        target = self.kinematics.pose_base(reference_q)
        if not np.allclose(pose(anchor),target,atol=2e-5,rtol=0):
            raise ValueError('chunk anchor不属于当前实际TCP，拒绝陈旧坐标参考')
        joint_chunk = np.zeros((16,8), dtype=np.float32);joint_chunk[:,7] = 1.
        self.last_targets = []
        for i in range(self.tcp_spec.execute_steps):
            target = apply_delta(target, physical[i,:6], anchor)
            next_q = self.kinematics.inverse(target, reference_q)
            joint_chunk[i,:7] = next_q-reference_q;joint_chunk[i,7] = physical[i,6]
            if np.any(np.abs(joint_chunk[i,:7]) > self.joint.action_adapter.delta_limits+1e-7):
                raise ValueError('TCP动作对应的关节增量过大，拒绝而不改写动作')
            self.last_targets.append(dict(base_from_tcp=target.tolist(),q=next_q.tolist()))
            reference_q = next_q
        return self.joint.execute(joint_chunk,controller)
