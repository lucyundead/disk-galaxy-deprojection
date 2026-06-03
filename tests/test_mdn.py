import torch

from dgdp.models.mdn import SummaryResidualMDN


def test_mdn_outputs_distribution_shapes():
    model = SummaryResidualMDN(input_dim=10, output_dim=4, hidden_dim=16, n_components=3)
    logits, means, log_scales = model(torch.zeros(5, 10))

    assert logits.shape == (5, 3)
    assert means.shape == (5, 3, 4)
    assert log_scales.shape == (5, 3, 4)


def test_mdn_negative_log_likelihood_is_finite():
    model = SummaryResidualMDN(input_dim=10, output_dim=4, hidden_dim=16, n_components=2)
    x = torch.zeros(5, 10)
    y = torch.zeros(5, 4)

    loss = model.negative_log_likelihood(x, y)

    assert torch.isfinite(loss)
