from types import SimpleNamespace

import numpy as np

from decision4rec.operators.fusion import choice_data, fit_conditional_logit, leave_one_domain_out, shrunk_weights


def choices(b, n=4000, K=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, K, len(b)))
    u = X @ np.asarray(b) + rng.gumbel(size=(n, K))
    return X, u.argmax(1)


def test_conditional_logit_recovers_the_weights():
    b = fit_conditional_logit(*choices([1.0, 0.5]))
    assert np.abs(b - [1.0, 0.5]).max() < 0.1


def test_scale_of_a_term_only_rescales_its_weight():
    X, y = choices([1.0, 0.5])
    a = fit_conditional_logit(X, y, l2=0.0)
    X2 = X.copy()
    X2[..., 1] *= 2.0
    c = fit_conditional_logit(X2, y, l2=0.0)
    assert abs(c[1] - a[1] / 2) < 1e-4 and abs(c[0] - a[0]) < 1e-4


def test_a_domain_takes_weights_fitted_on_the_others_only():
    data = {"a": choices([1.0, 0.5], seed=1), "b": choices([1.0, 0.5], seed=2), "c": choices([0.2, 2.0], seed=3)}
    w = leave_one_domain_out(data)
    changed = dict(data, a=choices([5.0, 0.0], seed=4))
    assert np.allclose(w["a"], leave_one_domain_out(changed)["a"])
    assert not np.allclose(w["b"], leave_one_domain_out(changed)["b"])


def test_choice_data_keeps_users_with_the_target_among_the_most_common_size():
    case = lambda k, pos: SimpleNamespace(cands=list(range(k)), target_pos=pos)
    cases = {0: case(3, 1), 1: case(3, -1), 2: case(3, 2), 3: case(2, 0)}
    Z = {c: np.zeros((len(cases[c].cands), 2)) for c in cases}
    X, y = choice_data(cases, Z)
    assert X.shape == (2, 3, 2) and y.tolist() == [1, 2]


def test_each_domain_counts_the_same_and_a_family_is_not_fitted_on_itself():
    big, small = choices([1.0, 0.0], n=4000, seed=5), choices([0.0, 1.0], n=500, seed=6)
    data = {"a": choices([1.0, 1.0], seed=7), "big": big, "small": small}
    b = leave_one_domain_out(data)["a"]
    assert abs(b[0] - b[1]) < 0.15                 # not pulled to the domain with more users
    fam = {"m1": "m", "m2": "m"}
    data = {"m1": choices([1.0, 0.0], seed=8), "m2": choices([0.0, 1.0], seed=9), "z": choices([0.5, 0.5], seed=10)}
    w = leave_one_domain_out(data, fam)
    assert np.allclose(w["m1"], w["m2"])           # both take the weights of z alone
    assert np.allclose(w["m1"], fit_conditional_logit(*data["z"]))


def test_a_small_domain_stays_near_the_others_and_a_large_one_moves_to_its_own():
    data = {"a": choices([1.0, 0.0], seed=11), "b": choices([1.0, 0.0], seed=12), "few": choices([0.0, 1.0], n=30, seed=13),
            "many": choices([0.0, 1.0], n=4000, seed=14)}
    w = shrunk_weights(data, strength=200.0)
    assert w["few"][0] > w["few"][1]               # the prior wins
    assert w["many"][1] > 3 * w["many"][0]         # its own choices win
