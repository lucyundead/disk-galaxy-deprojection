import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dgdp.model import _relu  # noqa: E402
from dgdp.models.mdn import SummaryResidualMDN  # noqa: E402


def test_numpy_meanhead_matches_torch():
    d_in, hid, out = 20, 128, 8
    mdn = SummaryResidualMDN(d_in, out, hidden_dim=hid, n_components=1).eval()
    x = torch.randn(5, d_in)
    with torch.no_grad():
        _, means, _ = mdn.forward(x)                         # (5,1,out) deterministic mean head
    W1 = mdn.net[0].weight.detach().numpy()
    b1 = mdn.net[0].bias.detach().numpy()
    W2 = mdn.net[2].weight.detach().numpy()
    b2 = mdn.net[2].bias.detach().numpy()
    Wm = mdn.means.weight.detach().numpy()
    bm = mdn.means.bias.detach().numpy()
    xn = x.numpy()
    h = _relu(_relu(xn @ W1.T + b1) @ W2.T + b2)
    got = h @ Wm.T + bm
    np.testing.assert_allclose(got, means[:, 0, :].numpy(), rtol=1e-5, atol=1e-5)
