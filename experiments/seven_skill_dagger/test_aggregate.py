import json
import pytest
from experiments.seven_skill_dagger.aggregate import aggregate


def part(root, seed):
    root.mkdir()
    record=dict(seed=seed,skill=0,case='until-stop',split='train',status='no_candidate',success=False)
    (root/'collection.json').write_text(json.dumps(dict(status='completed',mode='collect',checkpoint_sha256='parent',
                                                     source_sha256='source',records=[record])))
    return root


def test_preserves_failed_units_and_source_identity(tmp_path):
    a,b=part(tmp_path/'a',1),part(tmp_path/'b',2)
    result=aggregate([a,b],tmp_path/'merged',0,'parent')
    assert len(result['records'])==2
    assert all(r['status']=='no_candidate' for r in result['records'])
    assert len(result['sources'])==2


def test_rejects_duplicate_before_creating_output(tmp_path):
    a,b=part(tmp_path/'a',1),part(tmp_path/'b',1)
    with pytest.raises(ValueError):aggregate([a,b],tmp_path/'merged',0,'parent')
    assert not (tmp_path/'merged').exists()
