"""相同seed不能替代实际初态、首帧和Memory的一致性核验。"""
from copy import deepcopy

import pytest

from experiments.rgbd_memory_policy.evaluate import verify_initial_pair


@pytest.mark.parametrize('field',['input_digest','q','tcp','object_world','snapshot'])
def test_initial_pair_rejects_changed_evidence(field):
    initial=dict(input_digest='same',q=[0.]*7,tcp=[[1.]],object_world=[[1.]],snapshot={'available':True,'features':[.1]*12})
    verify_initial_pair(initial,deepcopy(initial))
    changed=deepcopy(initial);changed[field]='different'
    with pytest.raises(ValueError,match='未配对'):verify_initial_pair(initial,changed)
