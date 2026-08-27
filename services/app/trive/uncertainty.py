"""
Delta-method 1-sigma bands for a fitted Prony series: the sigma-counterpart of
`prony.compute_complex` / `prony.compute_relaxation_modulus`.

Everything here is a pure function of (tau_i, E_i, covariance) — no fitting, no
caching, no plotting. `fit` produces the coefficients and `quality` the Laplace
posterior covariance these functions consume; `figures` turns their output into
band traces.

Semantics of every number produced here (see the _FitQuality header in
`quality` for the covariance itself): +/-1 sigma CREDIBLE intervals of the
Laplace posterior over log-coefficients, conditional on the error inputs and
the smoothness setting. Smoothing bias is not in the covariance, so at strong
smoothing these bands undercover the truth read as frequentist error bars;
outside the data window they reflect the smoothness prior only — which is
exactly why they widen there.
"""

import numpy as np

from .prony import prony_basis


# Ceiling on any DISPLAYED sigma of a log-coefficient, in nepers (~6 decades).
# Where the data does not constrain a term at all — weak smoothing on the
# extended tail — the posterior sigma legitimately explodes; displaying it raw
# would overflow exp(), emit inf into stdlib-json (which raises), and detonate
# plotly's autorange. Six decades already reads as "unconstrained" on every
# plot while keeping all downstream numbers finite.
_SIGMA_DISPLAY_CAP = float(np.log(1e6))


def _split_terms(tau_i: np.ndarray, E_i: np.ndarray,
                 covariance: np.ndarray) -> tuple:
    """
    Validate the (tau_i, E_i, covariance) shape contract; split off the plateau.

    The contract (established by smooth_prony_fit): covariance rows follow the
    log-coefficients the solver actually searched, so its order is
    [log E_eq, log E_1..N] when the equilibrium modulus was a free parameter
    (interior solid path) and [log E_1..N] when it was not (solid=False, and
    the clamped path where E_eq pinned at exactly 0 — log(0) has no curvature
    to invert).

    Parameters:
        tau_i (numpy.ndarray): 1-D array of N relaxation times.
        E_i (numpy.ndarray): 1-D coefficient array of length N (viscous) or
            N + 1 (solid, equilibrium term first).
        covariance (numpy.ndarray): 2-D posterior covariance, (N, N) or
            (N + 1, N + 1) per the contract above.

    Returns:
        tuple: (E_terms, has_eq_row) — the N decaying coefficients, and whether
        covariance carries the equilibrium row (True only when E_i has the
        extra leading term AND covariance matches its full length).

    Raises:
        ValueError: On any shape combination outside the contract.
    """
    N = len(tau_i)
    solid = len(E_i) - N
    if solid not in (0, 1):
        raise ValueError(
            f"E_i has {len(E_i)} entries for {N} relaxation times; expected "
            f"{N} or {N + 1}"
        )
    cov = np.asarray(covariance)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square, got shape {cov.shape}")
    if solid and cov.shape[0] == N + 1:
        return E_i[1:], True
    if cov.shape[0] == N:
        return E_i[solid:], False
    raise ValueError(
        f"covariance has {cov.shape[0]} rows for {N} relaxation times "
        f"and {len(E_i)} coefficients; expected {N}"
        + (f" or {N + 1}" if solid else "")
    )


def sigma_log_coefficients(covariance: np.ndarray) -> np.ndarray:
    """
    Posterior 1-sigma of each log-coefficient: sqrt of the covariance diagonal.

    Row order is the covariance's own (see _split_terms). Uncapped — callers
    that display these apply _SIGMA_DISPLAY_CAP themselves.

    Parameters:
        covariance (numpy.ndarray): 2-D posterior covariance of the
            log-coefficients.

    Returns:
        numpy.ndarray: 1-D array of standard deviations, in nepers.
    """
    return np.sqrt(np.diag(covariance))


def spectrum_error_bars(E_terms: np.ndarray, sigma_log: np.ndarray,
                        cap: float = _SIGMA_DISPLAY_CAP) -> tuple:
    """
    Asymmetric LINEAR error bars for coefficients with log-normal uncertainty.

    A 1-sigma interval on log E is [E * exp(-s), E * exp(s)]; as offsets from
    E that is +E * expm1(s) upward and -E * (-expm1(-s)) downward — asymmetric
    in linear units, deliberately, since plotly error bars are drawn linearly
    even on log axes. expm1 keeps tiny sigmas exact, and the lower offset can
    never exceed E, so a capped bar stays positive on a log axis.

    Parameters:
        E_terms (numpy.ndarray): 1-D array of (non-negative) coefficients.
        sigma_log (numpy.ndarray): 1-D array of log-space sigmas, same length.
        cap (float): Ceiling applied to sigma_log before exponentiating; see
            _SIGMA_DISPLAY_CAP.

    Returns:
        tuple: (plus, minus) — 1-D non-negative offset arrays, upward and
        downward, in modulus units.
    """
    s = np.minimum(sigma_log, cap)
    return E_terms * np.expm1(s), E_terms * (-np.expm1(-s))


def complex_modulus_sigma(omega: np.ndarray, tau_i: np.ndarray,
                          E_i: np.ndarray, covariance: np.ndarray) -> dict:
    """
    1-sigma of the reconstructed storage/loss moduli and tan delta over omega.

    Delta method on the log-coefficients x = log(c): the model is linear in
    c = exp(x), so dy/dx_j = b_j * c_j with b_j the basis column, and
    var(y) = g.T @ Sigma @ g. Terms the data cannot pin get a huge log-sigma
    times a negligible weight c_j — no special-casing needed.

    tan delta = E''/E' is propagated DIRECTLY through the ratio,
    g_tan = g''/E' - (E''/E'**2) * g', never by combining separately-computed
    E' and E'' sigmas: the two moduli share every coefficient, and that
    correlation makes the true band much narrower than the uncorrelated
    combination would claim.

    Pass the SAME frequency grid the curves were evaluated on (e.g.
    compute_complex's own Frequency column) so band and curve cannot drift
    onto different grids.

    Parameters:
        omega (numpy.ndarray): 1-D array of angular frequencies to evaluate on.
        tau_i (numpy.ndarray): 1-D array of N relaxation times.
        E_i (numpy.ndarray): 1-D coefficient array, length N or N + 1
            (equilibrium term first).
        covariance (numpy.ndarray): 2-D posterior covariance per the
            _split_terms contract.

    Returns:
        dict: {'E Storage', 'E Loss', 'tan delta'} -> 1-D 1-sigma arrays over
        omega, in modulus units (dimensionless for tan delta).
    """
    E_terms, has_eq = _split_terms(tau_i, E_i, covariance)
    n = len(omega)
    # The curves themselves always include every coefficient E_i carries (a
    # clamped equilibrium term contributes exactly 0); the SENSITIVITY columns
    # follow the covariance's parameterization instead.
    full_basis = prony_basis(omega, tau_i, solid=len(E_i) > len(tau_i))
    curve = full_basis @ E_i
    E_stor, E_loss = curve[:n], curve[n:]

    params = E_i if has_eq else E_terms
    basis = full_basis if has_eq else full_basis[:, len(E_i) - len(tau_i):]
    G = basis * params  # dy/dx_j = b_j * c_j, columns scaled in place
    Gs, Gl = G[:n], G[n:]

    def _variance(rows):
        # einsum('ij,ij->i') is the row-wise quadratic form; clip the fp
        # negatives a PSD-in-exact-arithmetic product can produce.
        return np.einsum('ij,ij->i', rows @ covariance, rows).clip(min=0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        # E' can only be 0 if every coefficient is 0; guard anyway so a
        # degenerate row yields inf/nan variance that clip+sqrt surface as nan
        # rather than raising.
        G_tan = Gl / E_stor[:, None] \
            - (E_loss / E_stor ** 2)[:, None] * Gs
    return {
        'E Storage': np.sqrt(_variance(Gs)),
        'E Loss': np.sqrt(_variance(Gl)),
        'tan delta': np.sqrt(_variance(G_tan)),
    }


def relaxation_sigma(t: np.ndarray, tau_i: np.ndarray, E_i: np.ndarray,
                     covariance: np.ndarray) -> np.ndarray:
    """
    1-sigma of the decaying-terms-only relaxation modulus E(t) over t.

    Matches compute_relaxation_modulus, which EXCLUDES the equilibrium term
    from E(t) — so when the covariance carries an equilibrium row it is sliced
    off here. Dropping a row/column of a covariance matrix is exact
    marginalization over that parameter, not an approximation: E(t) simply
    does not depend on log E_eq.

    Parameters:
        t (numpy.ndarray): 1-D array of times to evaluate on — pass the same
            grid the curve was evaluated on.
        tau_i (numpy.ndarray): 1-D array of N relaxation times.
        E_i (numpy.ndarray): 1-D coefficient array, length N or N + 1.
        covariance (numpy.ndarray): 2-D posterior covariance per the
            _split_terms contract.

    Returns:
        numpy.ndarray: 1-D array of 1-sigma values over t, in modulus units.
    """
    E_terms, has_eq = _split_terms(tau_i, E_i, covariance)
    N = len(tau_i)
    cov = covariance[-N:, -N:] if has_eq else covariance
    G = np.exp(-np.outer(t, 1 / tau_i)) * E_terms
    return np.sqrt(np.einsum('ij,ij->i', G @ cov, G).clip(min=0.0))
