import numpy as np
from scipy import stats
from core import fitting as F, data as D, export as E


def test_exponential_recovered():
    x = stats.expon.rvs(scale=3.0, size=500, random_state=1)
    res = F.fit_all(x, ["expon", "norm", "uniform"])
    assert res[0].key == "expon"
    assert abs(res[0].params[1] - 3.0) < 0.4


def test_mle_matches_scipy_normal():
    x = stats.norm.rvs(10, 2, size=300, random_state=2)
    r = F.fit_one(x, "norm")
    loc, sc = stats.norm.fit(x)
    assert np.allclose(r.params, (loc, sc))
    assert np.isclose(r.aic, 2 * 2 - 2 * stats.norm.logpdf(x, loc, sc).sum())


def test_anderson_darling_matches_scipy():
    x = stats.norm.rvs(5, 1.5, size=200, random_state=3)
    dist = stats.norm
    _, ad = F._ks_ad(x, dist, (x.mean(), x.std(ddof=1)))
    assert np.isclose(ad, stats.anderson(x, "norm").statistic)


def test_ks_matches_scipy():
    x = stats.gamma.rvs(2.0, scale=2.0, size=300, random_state=4)
    r = F.fit_one(x, "gamma")
    assert np.isclose(r.ks, stats.kstest(x, "gamma", args=r.params).statistic)


def test_discrete_poisson():
    x = stats.poisson.rvs(4.0, size=400, random_state=5).astype(float)
    res = F.fit_all(x, ["poisson", "geom", "nbinom"])
    pois = next(r for r in res if r.key == "poisson")
    assert pois.chi2_p > 0.05          # no se rechaza Poisson
    assert "geom" not in [r.key for r in res]   # datos con ceros: geométrica (k>=1) se descarta


def test_shift_recovers_location():
    x = 5 + stats.expon.rvs(scale=2, size=600, random_state=6)
    r = F.fit_one(x, "expon", shift=True)
    assert abs(r.params[0] - 5) < 0.2


def test_bootstrap_pvalue_range():
    x = stats.norm.rvs(0, 1, size=80, random_state=7)
    r = F.fit_one(x, "norm")
    p_ks, p_ad = F.bootstrap_pvalues(x, r, n_boot=60)
    assert 0 < p_ks <= 1 and 0 < p_ad <= 1


def test_parse_text_and_runs():
    assert D.parse_text("1,5\n2,5\n3\n4\n5\n6\n7\n8").size == 8
    z, p = D.runs_test(np.random.default_rng(0).normal(size=200))
    assert 0 <= p <= 1


def test_export_expressions():
    r = F.fit_one(stats.expon.rvs(scale=2, size=200, random_state=8), "expon")
    assert E.arena_expr(r).startswith("EXPO(")
    assert "dist.rvs" in E.simpy_code(r)
