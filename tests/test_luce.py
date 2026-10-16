import numpy as np

from decision4rec.operators.design import components, random_partition
from decision4rec.operators.luce import Obs, fit_luce, fit_position_bias, kendall_tau


def _ask(block, theta, beta=None, out=None):
    z = theta[block] + (0 if beta is None else beta[: len(block)])
    if out is None:
        p = np.exp(z - z.max())
        return Obs(np.array(block), p / p.sum())
    zz = np.append(z, out)
    p = np.exp(zz - zz.max())
    p /= p.sum()
    return Obs(np.array(block), p[:-1], float(p[-1]))


def test_partition_covers_all_items_once():
    rng = np.random.default_rng(0)
    blocks = random_partition(range(23), 5, rng)
    assert sorted(i for b in blocks for i in b) == list(range(23))
    assert max(len(b) for b in blocks) <= 5 and min(len(b) for b in blocks) >= 4


def test_components_detects_disconnected_design():
    assert len(components([[0, 1], [2, 3]], 4)) == 2
    assert len(components([[0, 1], [2, 3], [1, 2]], 4)) == 1


def test_luce_recovers_utilities_from_consistent_answers():
    rng = np.random.default_rng(1)
    n = 30
    theta = rng.normal(0, 1.5, n)
    obs = [_ask(b, theta) for _ in range(3) for b in random_partition(range(n), 6, rng)]
    fit = fit_luce(obs, n, prior=1e-6, tol=1e-10, max_iter=20000)
    assert np.abs(fit.theta - (theta - theta.mean())).max() < 1e-3
    assert fit.residual < 1e-6
    assert kendall_tau(fit.theta, theta) > 0.999


def test_whole_pool_is_the_single_block_special_case():
    rng = np.random.default_rng(2)
    theta = rng.normal(0, 1, 12)
    fit = fit_luce([_ask(list(range(12)), theta)], 12)
    assert np.argsort(-fit.theta).tolist() == np.argsort(-theta).tolist()


def test_position_bias_is_recovered_and_removed():
    rng = np.random.default_rng(3)
    n, m = 40, 8
    beta = -0.4 * np.log1p(np.arange(m))
    groups, thetas = [], []
    for _ in range(40):
        th = rng.normal(0, 1.5, n)
        thetas.append(th)
        groups.append(([_ask(b, th, beta) for _ in range(2) for b in random_partition(range(n), m, rng)], n))
    est = fit_position_bias(groups, m)
    assert np.abs(est - beta).max() < 0.05
    obs, th = groups[0][0], thetas[0]
    err_raw = np.abs(fit_luce(obs, n).theta - (th - th.mean())).max()
    err_deb = np.abs(fit_luce(obs, n, beta=est).theta - (th - th.mean())).max()
    assert err_deb < err_raw and err_deb < 0.1


def test_outside_option_pool_probability():
    rng = np.random.default_rng(4)
    n = 24
    theta = rng.normal(0, 1, n)
    obs = [_ask(b, theta, out=0.7) for _ in range(3) for b in random_partition(range(n), 6, rng)]
    fit = fit_luce(obs, n, prior=1e-6, tol=1e-10, max_iter=20000)
    _, p_out = fit.pool_probs()
    truth = np.exp(0.7) / (np.exp(0.7) + np.exp(theta).sum())
    assert abs(p_out - truth) < 1e-3


def test_residual_grows_with_inconsistency():
    rng = np.random.default_rng(5)
    n = 20
    theta = rng.normal(0, 1, n)
    clean, noisy = [], []
    for _ in range(3):
        for b in random_partition(range(n), 5, rng):
            clean.append(_ask(b, theta))
            noisy.append(_ask(b, theta + rng.normal(0, 1.0, n)))
    assert fit_luce(noisy, n).residual > 10 * fit_luce(clean, n).residual
