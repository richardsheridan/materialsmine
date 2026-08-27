"""
Unit tests for app.trive.uncertainty — the delta-method 1-sigma bands.

Run from services/ with:  python -m unittest discover -s tests/trive -t .
"""

import unittest

import numpy as np

from app.trive.prony import prony_basis, compute_complex
from app.trive.fit import (
    smooth_prony_fit,
    _extended_relaxation_space,
    _GRID_EXTENSION_DECADES,
)
from app.trive.uncertainty import (
    _SIGMA_DISPLAY_CAP,
    _split_terms,
    sigma_log_coefficients,
    spectrum_error_bars,
    complex_modulus_noise,
    complex_modulus_sigma,
    relaxation_sigma,
)


def _random_cov(rng, m):
    """A well-conditioned random covariance: A @ A.T + a diagonal floor."""
    A = rng.normal(size=(m, m)) * 0.1
    return A @ A.T + 0.01 * np.eye(m)


class TestSplitTerms(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.tau_i = np.logspace(-2, 2, 5)

    def test_solid_with_equilibrium_row(self):
        E_i = np.abs(self.rng.normal(size=6)) + 0.1
        terms, has_eq = _split_terms(self.tau_i, E_i, _random_cov(self.rng, 6))
        np.testing.assert_array_equal(terms, E_i[1:])
        self.assertTrue(has_eq)

    def test_solid_clamped_covariance_has_no_equilibrium_row(self):
        E_i = np.concatenate(([0.0], np.abs(self.rng.normal(size=5)) + 0.1))
        terms, has_eq = _split_terms(self.tau_i, E_i, _random_cov(self.rng, 5))
        np.testing.assert_array_equal(terms, E_i[1:])
        self.assertFalse(has_eq)

    def test_viscous(self):
        E_i = np.abs(self.rng.normal(size=5)) + 0.1
        terms, has_eq = _split_terms(self.tau_i, E_i, _random_cov(self.rng, 5))
        np.testing.assert_array_equal(terms, E_i)
        self.assertFalse(has_eq)

    def test_rejects_off_contract_shapes(self):
        good_E = np.ones(6)
        with self.assertRaises(ValueError):  # E_i too long
            _split_terms(self.tau_i, np.ones(7), _random_cov(self.rng, 7))
        with self.assertRaises(ValueError):  # covariance size off-by-two
            _split_terms(self.tau_i, good_E, _random_cov(self.rng, 3))
        with self.assertRaises(ValueError):  # viscous E_i with (N+1)-cov
            _split_terms(self.tau_i, np.ones(5), _random_cov(self.rng, 6))
        with self.assertRaises(ValueError):  # non-square
            _split_terms(self.tau_i, good_E, np.ones((6, 5)))


class TestSpectrumErrorBars(unittest.TestCase):
    def test_formulas_and_asymmetry(self):
        E = np.array([2.0, 5.0])
        s = np.array([0.3, 1.0])
        plus, minus = spectrum_error_bars(E, s)
        np.testing.assert_allclose(plus, E * np.expm1(s))
        np.testing.assert_allclose(minus, E * (-np.expm1(-s)))
        # Log-symmetric interval: E + plus == E * e^s, E - minus == E * e^-s,
        # so the lower edge stays positive however large s gets.
        np.testing.assert_allclose(E + plus, E * np.exp(s))
        np.testing.assert_allclose(E - minus, E * np.exp(-s))
        self.assertTrue((minus < E).all())
        self.assertTrue((plus > minus).all())

    def test_display_cap_keeps_everything_finite(self):
        # sigma = 1000 nepers would overflow exp(); the cap turns it into the
        # six-decade "unconstrained" bar instead.
        E = np.array([1e9])
        plus, minus = spectrum_error_bars(E, np.array([1000.0]))
        self.assertTrue(np.isfinite(plus).all())
        np.testing.assert_allclose(plus, E * np.expm1(_SIGMA_DISPLAY_CAP))
        self.assertLess(minus[0], E[0])

    def test_sigma_log_coefficients_is_sqrt_diag(self):
        rng = np.random.default_rng(3)
        cov = _random_cov(rng, 4)
        np.testing.assert_allclose(
            sigma_log_coefficients(cov), np.sqrt(np.diag(cov)))


class TestDeltaMethod(unittest.TestCase):
    """complex_modulus_sigma / relaxation_sigma against brute force."""

    def setUp(self):
        self.rng = np.random.default_rng(21)
        self.tau_i = np.logspace(-2, 2, 6)
        self.omega = np.logspace(-2.5, 2.5, 15)
        self.E_i = np.abs(self.rng.normal(size=7)) + 0.5  # solid, eq first
        self.cov = _random_cov(self.rng, 7)

    def test_matches_per_point_quadratic_form(self):
        # var(y_i) = g_i.T @ Sigma @ g_i with g_ij = b_ij * c_j, spelled out
        # point by point with explicit loops.
        got = complex_modulus_sigma(self.omega, self.tau_i, self.E_i, self.cov)
        basis = prony_basis(self.omega, self.tau_i, solid=True)
        n = len(self.omega)
        for i in range(n):
            gs = basis[i] * self.E_i
            gl = basis[n + i] * self.E_i
            self.assertAlmostEqual(
                got['E Storage'][i], np.sqrt(gs @ self.cov @ gs), places=10)
            self.assertAlmostEqual(
                got['E Loss'][i], np.sqrt(gl @ self.cov @ gl), places=10)

    def test_tan_delta_matches_numerical_differentiation(self):
        # Central differences of tan delta as a function of x = log(c),
        # including the equilibrium coordinate — the direct ratio propagation
        # must agree with the chain rule done numerically.
        x = np.log(self.E_i)
        basis = prony_basis(self.omega, self.tau_i, solid=True)
        n = len(self.omega)

        def tan_delta(xv):
            curve = basis @ np.exp(xv)
            return curve[n:] / curve[:n]

        h = 1e-6
        grad = np.empty((n, len(x)))
        for j in range(len(x)):
            up, down = x.copy(), x.copy()
            up[j] += h
            down[j] -= h
            grad[:, j] = (tan_delta(up) - tan_delta(down)) / (2 * h)
        expected = np.sqrt(np.einsum('ij,ij->i', grad @ self.cov, grad))
        got = complex_modulus_sigma(self.omega, self.tau_i, self.E_i, self.cov)
        np.testing.assert_allclose(got['tan delta'], expected, rtol=1e-5)

    def test_tan_delta_narrower_than_uncorrelated_combination(self):
        # The reason for propagating the ratio directly: E' and E'' share
        # every coefficient. The uncorrelated quadrature combination
        # tan_d * sqrt((s'/E')**2 + (s''/E'')**2) must overstate the band.
        got = complex_modulus_sigma(self.omega, self.tau_i, self.E_i, self.cov)
        basis = prony_basis(self.omega, self.tau_i, solid=True)
        curve = basis @ self.E_i
        n = len(self.omega)
        E_stor, E_loss = curve[:n], curve[n:]
        uncorrelated = (E_loss / E_stor) * np.sqrt(
            (got['E Storage'] / E_stor) ** 2 + (got['E Loss'] / E_loss) ** 2)
        self.assertLess(np.median(got['tan delta'] / uncorrelated), 1.0)

    def test_relaxation_sigma_matches_brute_force_and_drops_equilibrium(self):
        t = np.logspace(-2, 2, 12)
        got = relaxation_sigma(t, self.tau_i, self.E_i, self.cov)
        # Brute force over the decaying block only, with the equilibrium
        # row/column marginalized out by slicing.
        cov_dec = self.cov[1:, 1:]
        for i, ti in enumerate(t):
            g = np.exp(-ti / self.tau_i) * self.E_i[1:]
            self.assertAlmostEqual(got[i], np.sqrt(g @ cov_dec @ g), places=10)
        # And identical to calling with the viscous parameterization directly.
        viscous = relaxation_sigma(t, self.tau_i, self.E_i[1:], cov_dec)
        np.testing.assert_allclose(got, viscous, rtol=1e-12)

    def test_clamped_parameterization(self):
        # E_i carries a (clamped, zero) equilibrium entry but the covariance
        # is decaying-only: sensitivities must align with the N columns.
        E_clamped = np.concatenate(([0.0], self.E_i[1:]))
        cov_dec = self.cov[1:, 1:]
        got = complex_modulus_sigma(
            self.omega, self.tau_i, E_clamped, cov_dec)
        basis = prony_basis(self.omega, self.tau_i, solid=True)[:, 1:]
        n = len(self.omega)
        gs = basis[:n] * self.E_i[1:]
        expected = np.sqrt(np.einsum('ij,ij->i', gs @ cov_dec, gs))
        np.testing.assert_allclose(got['E Storage'], expected, rtol=1e-12)


class TestComplexModulusNoise(unittest.TestCase):
    """The measurement-noise half of the prediction interval."""

    def setUp(self):
        self.rng = np.random.default_rng(31)
        self.tau_i = np.logspace(-2, 2, 6)
        self.E_i = np.abs(self.rng.normal(size=7)) + 0.5
        self.omega_data = np.logspace(-2, 2, 25)

    def _fit_curve(self, omega):
        basis = prony_basis(omega, self.tau_i, solid=True)
        curve = basis @ self.E_i
        n = len(omega)
        return curve[:n], curve[n:]

    def test_constant_relative_profile_inverts_exactly(self):
        # The relative-error setting: rel = const at every data point must
        # come back as sigma = rel * |E*_fit| on ANY evaluation grid — the
        # extension region included, where the profile is edge-clamped.
        rel = np.full(len(self.omega_data), 0.05)
        omega = np.logspace(-4, 4, 40)  # wider than the data window
        got = complex_modulus_noise(
            omega, self.tau_i, self.E_i, self.omega_data, rel, rel)
        stor, loss = self._fit_curve(omega)
        mag = np.abs(stor + 1.0j * loss)
        np.testing.assert_allclose(got['E Storage'], 0.05 * mag, rtol=1e-12)
        np.testing.assert_allclose(got['E Loss'], 0.05 * mag, rtol=1e-12)

    def test_profile_interpolates_in_log_frequency_and_clamps(self):
        # A profile that ramps across the window: evaluated at a data point
        # it returns that point's value; beyond the window it holds the edge
        # value (times the local fitted magnitude).
        rel = np.linspace(0.01, 0.10, len(self.omega_data))
        probe = np.array([1e-5, self.omega_data[7], 1e5])
        got = complex_modulus_noise(
            probe, self.tau_i, self.E_i, self.omega_data, rel, rel)
        stor, loss = self._fit_curve(probe)
        mag = np.abs(stor + 1.0j * loss)
        expected_rel = np.array([rel[0], rel[7], rel[-1]])
        np.testing.assert_allclose(
            got['E Storage'] / mag, expected_rel, rtol=1e-12)

    def test_unsorted_data_grid_is_handled(self):
        # chart passes the upload's own order; the interp must not require it
        # to be ascending.
        rel = np.linspace(0.01, 0.10, len(self.omega_data))
        perm = self.rng.permutation(len(self.omega_data))
        omega = np.logspace(-3, 3, 30)
        got = complex_modulus_noise(
            omega, self.tau_i, self.E_i, self.omega_data, rel, rel)
        shuffled = complex_modulus_noise(
            omega, self.tau_i, self.E_i,
            self.omega_data[perm], rel[perm], rel[perm])
        np.testing.assert_allclose(shuffled['E Storage'], got['E Storage'])

    def test_tan_delta_noise_is_the_independent_ratio_formula(self):
        # var(tan d) = (sigma''/E')**2 + (E'' sigma'/E'**2)**2 for
        # independent new readings of E' and E''.
        rel_s = np.full(len(self.omega_data), 0.03)
        rel_l = np.full(len(self.omega_data), 0.07)
        omega = np.logspace(-2, 2, 15)
        got = complex_modulus_noise(
            omega, self.tau_i, self.E_i, self.omega_data, rel_s, rel_l)
        stor, loss = self._fit_curve(omega)
        expected = np.sqrt(
            (got['E Loss'] / stor) ** 2
            + (loss * got['E Storage'] / stor ** 2) ** 2)
        np.testing.assert_allclose(got['tan delta'], expected, rtol=1e-12)

    def test_prediction_dominates_credible_in_quadrature(self):
        # The combination the figures draw: hypot(credible, noise) is at
        # least as large as either part, everywhere.
        rel = np.full(len(self.omega_data), 0.05)
        omega = np.logspace(-3, 3, 20)
        cov = self.rng.normal(size=(7, 7)) * 0.05
        cov = cov @ cov.T + 0.01 * np.eye(7)
        cred = complex_modulus_sigma(omega, self.tau_i, self.E_i, cov)
        data = complex_modulus_noise(
            omega, self.tau_i, self.E_i, self.omega_data, rel, rel)
        for key in cred:
            pred = np.hypot(cred[key], data[key])
            self.assertTrue((pred >= cred[key]).all())
            self.assertTrue((pred >= data[key]).all())


class TestBandsOnARealFit(unittest.TestCase):
    """Monotone growth of the band outside the data window, end to end."""

    @classmethod
    def setUpClass(cls):
        # A peaked master curve with 5% noise-free error bars; N well below
        # the span's rank so the fit is comfortable.
        tau = np.logspace(-4.0, 4.0, 9)
        E_input = np.concatenate(
            ([1e6], np.exp(-(np.log10(tau)) ** 2 / 4.0) * 1e9))
        df = compute_complex(tau, E_input, num_pts=200)
        cls.omega = df['Frequency'].to_numpy()
        E_stor = df['E Storage'].to_numpy()
        E_loss = df['E Loss'].to_numpy()
        std = np.abs(E_stor + 1.0j * E_loss) * 0.05
        cls.tau_i, cls.E_i, cls.quality = smooth_prony_fit(
            cls.omega, E_stor, E_loss, E_stor_std=std, E_loss_std=std,
            N=30, smoothness=0.1, solid=True, return_fit_quality=True,
        )
        _, cls.n_ext = _extended_relaxation_space(
            1 / cls.omega.max(), 1 / cls.omega.min(), 30,
            2 * len(cls.omega), True, _GRID_EXTENSION_DECADES,
        )

    def test_fixture_is_extended(self):
        self.assertGreater(self.n_ext, 0)
        self.assertIsNotNone(self.quality.covariance)

    def test_relative_curve_sigma_grows_off_both_window_edges(self):
        curve = compute_complex(self.tau_i, self.E_i)
        freq = curve['Frequency'].to_numpy()
        sig = complex_modulus_sigma(
            freq, self.tau_i, self.E_i, self.quality.covariance)
        rel = sig['E Storage'] / curve['E Storage'].to_numpy()
        inside = (freq >= self.omega.min()) & (freq <= self.omega.max())
        first_in, last_in = np.flatnonzero(inside)[[0, -1]]
        # The outermost extrapolated points must be less certain than the
        # nearest in-window ones — the band visibly widens off both edges.
        self.assertGreater(rel[0], rel[first_in])
        self.assertGreater(rel[-1], rel[last_in])
        # And the widening is substantial, not a rounding artifact. (Margin
        # calibrated to the 1-decade extension: the sigma growth is ~Delta^1.5
        # in the extrapolated distance, so a wider _GRID_EXTENSION_DECADES
        # only increases it.)
        self.assertGreater(rel[0], 1.5 * np.median(rel[inside]))
        self.assertGreater(rel[-1], 1.5 * np.median(rel[inside]))

    def test_tail_log_sigmas_exceed_the_window_median(self):
        slog = sigma_log_coefficients(self.quality.covariance)
        if self.quality.covariance.shape[0] == len(self.E_i):
            slog = slog[1:]  # drop the equilibrium row
        n_ext = self.n_ext
        window = slog[n_ext:len(slog) - n_ext]
        self.assertGreater(slog[:n_ext].max(), np.median(window))
        self.assertGreater(slog[-n_ext:].max(), np.median(window))
