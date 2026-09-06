"""采集 seed 不得因省略 callback 而获得模型执行权。"""
import pytest

from experiments.g2c_memory_integration.run import run


@pytest.mark.parametrize("kwargs", [
    {"development_seed":1000100, "vla_runtime":object()},
    {"development_seed":1000100, "vla_runtime":object(), "after_commit":lambda **_:None},
    {"after_commit":lambda **_:None},
    {"development_seed":1000112},
])
def test_invalid_collection_authority_rejected_before_output(tmp_path, kwargs):
    output = tmp_path / "route"
    with pytest.raises(ValueError):
        run(tmp_path / "missing-bundle", output, **kwargs)
    assert not output.exists()
