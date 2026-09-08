"""显式加载续训格式，保留原权重身份检查，不伪装成256步checkpoint。"""
import torch
from experiments.g2c_memory_integration.vla import sha256
from experiments.tcp_memory_control.protocol import identity

BEST_SHA = 'fd2d2a87e7a74619e666da321982324348bb7426fe2f132c7c4a1378752a19b0'
PARENT_SHA = 'eaf50c82b8a897261ecca1a25154e750bc23b477d0382c54d9c53023dd449a27'
CONFIG_SHA = '7dd80754fb14f3abdedc0ef01e0f128118ea63fb720e16598d87eafe25926dae'
EVALUATION_SHA = 'f5b4476b29dbb5ffbddc925ff0bc23b66f2b66592f7b4a7be12b39a5105bafa2'


def validate_payload(payload, config, parent_identity, original_source_hash):
    if (payload['format'] != 'tcp-teacher-continuation/v1'
        or payload['configuration_identity'] != identity(config)
        or payload['parent_sha256'] != PARENT_SHA
        or config['parent_sha256'] != PARENT_SHA
        or config['parent_identity'] != parent_identity
        or config['source_sha256'] != original_source_hash
        or payload['step'] != 58880 or payload['best_step'] != 58880):
        raise ValueError('不是本轮冻结的TCP最佳续训checkpoint')


def load_best(policy, path, config, parent_identity, original_source_hash):
    if sha256(path) != BEST_SHA:
        raise ValueError('续训权重SHA不匹配')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    validate_payload(payload, config, parent_identity, original_source_hash)
    policy.expert.load_state_dict(payload['expert'], strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'], strict=True)
    return dict(checkpoint_sha256=BEST_SHA, continuation_step=payload['step'],
                total_training_steps=payload['total_steps'])
