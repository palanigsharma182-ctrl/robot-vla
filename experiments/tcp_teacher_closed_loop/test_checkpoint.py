"""拒绝错误父模型、源码、选择点和旧格式；只核验本次身份边界。"""
import pytest
from experiments.tcp_memory_control.protocol import identity
from experiments.tcp_teacher_closed_loop.checkpoint import PARENT_SHA, validate_payload


def fixture():
    c=dict(parent_sha256=PARENT_SHA, parent_identity='parent', source_sha256='source')
    p=dict(format='tcp-teacher-continuation/v1', configuration_identity=identity(c),
           parent_sha256=PARENT_SHA,step=58880,best_step=58880)
    return p,c


def test_expected_identity_passes():
    p,c=fixture()
    validate_payload(p,c,'parent','source')


@pytest.mark.parametrize('field,value', [('format','tcp-memory-combined-checkpoint/v1'),
    ('parent_sha256','wrong'),('configuration_identity','wrong'),('step',59877),('best_step',0)])
def test_changed_payload_rejected(field,value):
    p,c=fixture();p[field]=value
    with pytest.raises(ValueError):validate_payload(p,c,'parent','source')


@pytest.mark.parametrize('parent,source',[('other','source'),('parent','other')])
def test_wrong_parent_contract_rejected(parent,source):
    p,c=fixture()
    with pytest.raises(ValueError):validate_payload(p,c,parent,source)


def test_protocol_roundtrip_preserves_identity_but_changed_seed_does_not():
    import json
    protocol=dict(arms=('before','after'),seeds=[1600200,1600201])
    decoded=json.loads(json.dumps(protocol))
    assert identity(decoded)==identity(protocol)
    decoded['seeds'][0]=1600999
    assert identity(decoded)!=identity(protocol)
