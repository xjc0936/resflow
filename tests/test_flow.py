import torch

from resflow.flow import ResFlowProcess


def test_paper_schedule_endpoints_and_derivative():
    process = ResFlowProcess(beta=10, gamma=1.75)
    t = torch.tensor([0.0, 1.0])
    assert torch.allclose(process.sigma_y(t), torch.tensor([10 / 11, 1.0]))
    assert torch.allclose(process.sigma_y_derivative(t), torch.tensor([10 / 121, 0.1]))


def test_training_targets():
    process = ResFlowProcess()
    hq = torch.zeros(2, 3, 4, 4)
    lq = torch.ones_like(hq)
    noise = torch.ones_like(hq)
    batch = process.training_batch(hq, lq, t=torch.tensor([0.0, 1.0]), y1=noise)
    assert batch.target.shape == (2, 6, 4, 4)
    assert torch.allclose(batch.target[:, :3], torch.ones_like(hq))
    assert batch.weight[0] == 0
    assert torch.allclose(batch.weight[1], torch.tensor(1.0))


class ConstantVelocity(torch.nn.Module):
    def forward(self, x, y, t):
        return torch.cat((torch.ones_like(x), torch.zeros_like(y)), dim=1)


def test_reverse_euler_sign():
    process = ResFlowProcess()
    lq = torch.ones(1, 3, 4, 4)
    result = process.restore(ConstantVelocity(), lq, steps=4, y1=torch.zeros_like(lq), clamp=False)
    assert torch.allclose(result, torch.zeros_like(lq))

