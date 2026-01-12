import numpy as np

from group_elephant.workstreams.data import secom_preprocess as sp


def test_drop_high_missing_columns():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 4.0]])
    names = ["a", "b"]
    X2, n2, mask = sp.drop_high_missing_columns(X, names, max_missing_fraction=0.6)
    assert n2 == ["a"]
    assert X2.shape == (3, 1)
    assert mask.tolist() == [True, False]


def test_impute_missing_median():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    X2, vals = sp.impute_missing(X, "median")
    assert not np.isnan(X2).any()
    assert vals.shape == (2,)
    assert X2[0, 1] == 5.0


def test_drop_constant_features():
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    names = ["const", "var"]
    X2, n2, mask = sp.drop_constant_features(X, names)
    assert n2 == ["var"]
    assert X2.shape == (3, 1)
    assert mask.tolist() == [False, True]


def test_standardize():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xs, mu, sigma = sp.standardize(X)
    assert np.allclose(mu, [2.0, 3.0])
    assert np.allclose(sigma, [1.0, 1.0])
    assert np.allclose(np.mean(Xs, axis=0), [0.0, 0.0])
