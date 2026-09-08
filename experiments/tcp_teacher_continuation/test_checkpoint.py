"""完整checkpoint重载后的下一步必须重现不中断训练。"""
import copy
import numpy as np
import torch
from experiments.tcp_teacher_continuation.run import atomic_checkpoint


def test_optimizer_and_random_state_resume(tmp_path):
    torch.manual_seed(42)
    rng = np.random.default_rng(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def update():
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(2,3) * float(rng.random())
        model(x).square().mean().backward()
        optimizer.step()

    update()
    path = tmp_path/'latest.pt'
    atomic_checkpoint(path, dict(model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()), numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state()))
    update()
    expected = copy.deepcopy(model.state_dict())
    p = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(p['model'])
    optimizer.load_state_dict(p['optimizer'])
    rng.bit_generator.state = p['numpy_rng']
    torch.set_rng_state(p['torch_rng'])
    update()
    assert all(torch.equal(v, expected[k]) for k,v in model.state_dict().items())
    assert not path.with_suffix('.tmp').exists()
