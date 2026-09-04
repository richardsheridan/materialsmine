"""
The chunked QR reduction that makes the fit's cost independent of upload size,
the content-addressed LRU cache that lets a slider sweep reuse it, and the two
probes that read term-count limits off one reduced triangle: the numerical
and statistical ceilings (`prony_rank_limits`) and the data's resolution at
the current smoothing (`prony_resolution`).

This is the module the fit's performance rests on: it turns an
O(rows) x O(N) weighted least-squares problem into an (N + 2) x (N + 1)
triangle, exactly. See `_prony_reduce` for why the residual row is kept, and the
project README/CLAUDE notes for the benchmark that says not to "simplify" it
away.
"""

import hashlib
from collections import OrderedDict

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import nnls

from .objective import _add_penalty_inplace, _penalty_trace, _scaled_smoothness
from .prony import prony_basis, prony_relaxation_space, PRONY_TERMS_MAX


# Frequency points per block in smooth_prony_fit's chunked QR reduction. Each
# block materializes a (2 * chunk, N + 2) basis slab (plus prony_basis's single
# reciprocal temporary), so peak memory is O(chunk * N) no matter how many
# rows the upload has.
_QR_CHUNK_ROWS = 8192

# Reduced systems retained by _prony_reduce's LRU cache. Each entry holds only
# the (m + 1) x (m + 1) triangle — a couple hundred rows at the extreme, since
# the route caps user N at 100, the rank probe adds 8, and the smoothed path's
# grid extension can add up to dof // 2 tail terms (fit._GRID_EXTENSION_DECADES)
# — so the cache stays tiny no matter how large the uploads that produced it.
# Sized for a smoothness sweep, which varies only `smoothness` and can reuse
# one reduction throughout. Each dataset can fill THREE slots — the smoothed
# fit's extended grid, the smoothness == 0 fit's unextended grid, and the
# fixed probe grid shared by prony_rank_limits and prony_resolution (one
# slot, both hit it) — so 6 keeps one dataset's sweep resident with room for
# a second dataset.
_REDUCE_CACHE_SIZE = 6


# digest -> (R, z), least-recently-used first. See _prony_reduce.
_REDUCE_CACHE = OrderedDict()


def _reduce_cache_key(arrays: tuple, solid: bool) -> tuple:
    """
    Content-address the reduction inputs without retaining them.

    Returns a hashable key holding only a digest, never the arrays themselves,
    so a cache entry cannot pin a 40,000-row upload in memory. Hashing ~1 MB
    costs on the order of a millisecond against seconds for the QR it saves.

    Parameters:
        arrays (tuple): numpy arrays the reduction depends on.
        solid (bool): Whether an equilibrium term is included.

    Returns:
        tuple: (digest bytes, solid) — hashable and content-addressed.
    """
    digest = hashlib.blake2b(digest_size=16)
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.shape).encode())
        digest.update(arr.dtype.str.encode())
        digest.update(memoryview(arr).cast('B'))  # no copy for contiguous input
    return digest.digest(), bool(solid)


def _prony_reduce(
        omega: np.ndarray,
        E_stor: np.ndarray,
        E_loss: np.ndarray,
        E_stor_std: np.ndarray,
        E_loss_std: np.ndarray,
        tau_i: np.ndarray,
        solid: bool,
        std_scale: float = 1.0,
) -> tuple:
    """
    Compress the weighted least-squares problem by a chunked QR factorization.

    Maintains the triangular augmented system [R | z] and folds each weighted
    basis block into it, so that EXACTLY
        ||(y - B c) / (std_scale * std)||^2 = ||R c - z||^2
    and the reduced system has at most len(tau_i) + solid + 1 rows regardless of
    how many data rows the upload carries. Householder QR accumulates the residual
    information backward-stably (no explicit sums of squares), memory stays
    O(_QR_CHUNK_ROWS * N), and every subsequent solver operation costs O(N^2)
    independent of the input row count.

    Memoized on input CONTENT in a size-_REDUCE_CACHE_SIZE LRU, because a
    smoothness sweep varies only `smoothness` — which this reduction does not
    depend on — and would otherwise redo the expensive pass over every row for
    each trial value. Cached R and z are returned READ-ONLY: scipy 1.10's nnls
    and minimize do not write to them, but a future scipy that did would
    otherwise silently poison every later cache hit. The cache is not
    synchronized; under gunicorn's sync workers only one request runs per
    process, and a threaded worker could at worst duplicate work or perturb LRU
    order, never corrupt an entry.

    std_scale is a UNIFORM multiplier on both std arrays, held out of the QR and
    out of the cache key: R and z are proportional to 1/std, so it divides back
    out of the (m + 1) x (m + 1) result and the O(rows) pass never sees it. That
    is what makes the relative-error widget cheap — varying only this factor
    reuses one reduction. It divides on every call, so hits and misses match.

    Parameters:
        omega (numpy.ndarray): 1-D array of angular frequencies.
        E_stor (numpy.ndarray): 1-D array of storage-modulus values.
        E_loss (numpy.ndarray): 1-D array of loss-modulus values.
        E_stor_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_stor.
        E_loss_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_loss.
        tau_i (numpy.ndarray): 1-D relaxation-time grid.
        solid (bool): Whether to include an equilibrium-modulus term.
        std_scale (float): Uniform positive multiplier on both std arrays, kept
            out of the reduction and divided out of the result.

    Returns:
        tuple: (R, z), the reduced design matrix and target. The equality above
        is exact with no correction term to carry: see the comment on the
        residual row below. Both are fresh arrays scaled from the read-only
        cache entry, so a consumer that wrote to them could not poison it.
    """
    key = _reduce_cache_key(
        (omega, E_stor, E_loss, E_stor_std, E_loss_std, tau_i), solid
    )
    hit = _REDUCE_CACHE.get(key)
    if hit is not None:
        _REDUCE_CACHE.move_to_end(key)
        return hit[0] / std_scale, hit[1] / std_scale

    m = len(tau_i) + solid
    # Keeping m + 1 rows retains the full least-squares information. Row m of
    # the final triangle carries the orthogonal residual the fit can never
    # reach, and it is KEPT in (R, z) rather than returned as a separate
    # constant: R is upper triangular, so that row is exactly zero across the
    # basis columns, making it an ordinary residual row that contributes
    # z[m]**2 to any loss and nothing at all to any gradient. Every consumer
    # therefore sees full-problem values with no offset to thread through, and
    # the reduction stays exact rather than exact-up-to-a-correction. For
    # uploads with fewer than m rows the triangle is simply shorter (wide R) —
    # nnls and _prony_objective both accept that shape, and there is then no
    # unreachable residual to carry.
    Rz = np.empty((0, m + 1))
    for start in range(0, len(omega), _QR_CHUNK_ROWS):
        chunk = slice(start, start + _QR_CHUNK_ROWS)
        basis = prony_basis(omega[chunk], tau_i, solid)
        y = np.concatenate((E_stor[chunk], E_loss[chunk]))
        y_std = np.concatenate((E_stor_std[chunk], E_loss_std[chunk]))
        block = np.concatenate(
            (basis / y_std[:, None], (y / y_std)[:, None]), axis=1
        )
        Rz = np.linalg.qr(
            np.concatenate((Rz, block), axis=0), mode='r'
        )[:m + 1]
    Rz.flags.writeable = False
    reduced = (Rz[:, :m], Rz[:, m])

    _REDUCE_CACHE[key] = reduced
    if len(_REDUCE_CACHE) > _REDUCE_CACHE_SIZE:
        _REDUCE_CACHE.popitem(last=False)
    return reduced[0] / std_scale, reduced[1] / std_scale


# Probe grid size for prony_rank_limits: enough columns that the measured rank
# is never truncated by the grid right at the PRONY_TERMS_MAX ceiling. The
# sqrt(eps)-rank density of this basis is ~7.5 columns/decade at float64 (see
# the rank law in prony_rank_limits), so the counts that matter — those at or
# below the route's cap of 100 — are only reachable for spans under ~12.5
# decades, where this grid provides >= ~8.6 columns/decade: denser than the
# rank it is measuring (and near that crossover any residual grid-truncation
# errs low, i.e. toward a slightly lower cap).
_RANK_PROBE_TERMS = PRONY_TERMS_MAX + 8


def _probe_grid(omega: np.ndarray) -> np.ndarray:
    """
    The fixed probe grid: _RANK_PROBE_TERMS log-spaced tau over the data
    window. One function so prony_rank_limits and prony_resolution hash the
    byte-identical array into the same _prony_reduce cache entry.
    """
    return prony_relaxation_space(
        1 / np.max(omega), 1 / np.min(omega), _RANK_PROBE_TERMS
    )


def prony_rank_limits(
        omega: np.ndarray,
        E_stor: np.ndarray,
        E_loss: np.ndarray,
        E_stor_std: np.ndarray,
        E_loss_std: np.ndarray,
        solid: bool = True,
        std_scale: float = 1.0,
) -> tuple:
    """
    How many Prony terms this dataset can determine: (max_prony, noise_prony).

    The Prony basis over D decades of frequency has numerical rank
    ~ 0.47 * D * ln(1/eps) + O(1), INDEPENDENT of the term count: its singular
    values decay geometrically, sigma_k ~ exp(-pi**2 k / (2 D ln 10)) — the
    classical inverse-Laplace ill-posedness, halved in rate because the basis
    stacks two kernel families (storage and loss). Because the rank is a
    property of the span and the precision, not of the grid, ONE fixed
    overcomplete probe grid (_probe_grid: _RANK_PROBE_TERMS log-spaced tau
    over the data's own span) measures it for every possible N, and one SVD
    of the QR-reduced triangle — at most ~110 x 110 whatever the upload size
    — reads it off. The probe reduction content-addresses to its own
    _prony_reduce cache entry, distinct from any fit's and shared with
    prony_resolution, so repeated calls on one dataset pay the O(rows) pass
    once; and since std_scale is excluded from that key, error-widget sweeps
    redo only the SVD.

    Two counts come back, for two different consumers:

      * max_prony — the slider's ceiling: singular values above
        sqrt(eps) * sigma_max, where eps is the machine epsilon OF THE INPUT
        ARRAYS' dtype (np.result_type of the two modulus arrays). sqrt, not
        eps itself, because what the fit actually factors is the Gram
        R.T @ R (the Gauss-Newton block of the Newton Hessian, and of the
        Laplace covariance), whose eigenvalues are the SQUARED singular
        values — so sqrt(eps) on the basis is exactly eps on the Gram, and
        this count is the float64 numerical rank of the Gram at the flat
        seed. Terms past it are columns the Gram cannot tell apart from
        combinations of the others: the data has no say in them, the
        smoothness prior alone fills them, and a finer grid changes what the
        fit reports outside the data, not what it resolves inside it —
        harmless (the penalty regularizes them completely; the production
        grid extension already runs every fit well past this rank) but
        pointless, which is why the slider stops offering them. It is NOT a
        solver-liveness guard: the trust-exact hang that was once blamed on
        N past this rank reproduces well below it (fit._newton_watchdog has
        the mechanism, fit._NEWTON_HESSIAN_SHIFT_EPS the fix). Everything
        upstream computes in float64 today, but data that arrives as float32
        carries only float32 information — modes below its quantization
        floor would fit rounding noise — so the ceiling follows the data's
        own precision with no code change if a lower-precision source ever
        appears. Measured through this probe at float64 the count runs
        ~7.5 * D + 7.5 over D decades. Uniform std_scale moves every sigma
        together and cancels out of this count; the SHAPE of the std
        profile does move it, deliberately — the weighted system is the one
        actually solved. Clipped to [1, PRONY_TERMS_MAX] after dropping the
        equilibrium column (a determined plateau is not a relaxation term
        the slider counts).

      * noise_prony — the statistical ceiling under the SELECTED error model
        and the WEAKEST scale-respecting prior. _prony_reduce returns the
        triangle already divided by (std * std_scale), so weighted noise has
        unit variance and singular direction k of the coefficient vector is
        determined to a standard deviation of 1/sigma_k in modulus units;
        it counts as data-determined when that beats the modulus scale:
        sigma_k * max(E_stor) > 1. Measured 2026-09-04 on every bundled
        master curve, this count equals, to within 0.3, MacKay's effective
        number of well-determined parameters
            gamma = sum_k sigma_k**2 E_max**2 / (sigma_k**2 E_max**2 + 1)
        under an isotropic Gaussian prior of width max(E_stor) on every
        coefficient — the weakest prior that respects the physical scale
        (any coefficient could be anywhere up to the whole modulus). It is
        therefore a smoothness-free CEILING on what any smoothing prior can
        leave to the data: a smoothness prior is more informative than that
        box in every high-curvature direction, so on real (smooth) spectra
        the smoothed fit's actual resolution (prony_resolution) sits 2-4x
        below it; and it moves only ~1.07 * D directions per decade of
        assumed error — a decade of tighter error buys about one direction
        per decade of span. A COUNT of singular directions, not a roster of
        terms: no particular Prony term is the identifiable one. Kept on the
        wire for clients; it no longer drives a caption — the grid-size
        suggestion measures against prony_resolution on both fit paths.
        Unclipped and including the equilibrium column.

    Parameters:
        omega (numpy.ndarray): 1-D array of angular frequencies.
        E_stor (numpy.ndarray): 1-D array of storage-modulus values.
        E_loss (numpy.ndarray): 1-D array of loss-modulus values.
        E_stor_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_stor.
        E_loss_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_loss.
        solid (bool): Whether the probe includes an equilibrium-modulus column,
            matching the fit the counts will be compared against.
        std_scale (float): Uniform positive multiplier on both std arrays,
            exactly as passed to smooth_prony_fit.

    Returns:
        tuple: (max_prony, noise_prony) as plain Python ints — the route
        serializes with stdlib json.dumps, which rejects numpy scalars.
    """
    tau_probe = _probe_grid(omega)
    R, _ = _prony_reduce(
        omega, E_stor, E_loss, E_stor_std, E_loss_std,
        tau_probe, solid, std_scale,
    )
    sigma = np.linalg.svd(R, compute_uv=False)
    eps = np.finfo(np.result_type(E_stor, E_loss)).eps
    # sqrt: the Gram, not the basis, is what gets factored (see docstring).
    eps_count = int(np.count_nonzero(sigma > np.sqrt(eps) * sigma[0]))
    max_prony = int(min(PRONY_TERMS_MAX, max(1, eps_count - solid)))
    noise_prony = int(np.count_nonzero(sigma * np.max(E_stor) > 1.0))
    return max_prony, noise_prony


def prony_resolution(
        omega: np.ndarray,
        E_stor: np.ndarray,
        E_loss: np.ndarray,
        E_stor_std: np.ndarray,
        E_loss_std: np.ndarray,
        tau_i: np.ndarray,
        E_i: np.ndarray,
        smoothness: float,
        solid: bool = True,
        std_scale: float = 1.0,
):
    """
    How many relaxation terms a DENSE grid would resolve from this data at
    this smoothing — "the data's resolution" the grid-size suggestion is
    measured against. A float, or None when undefined.

    Two definitions, one per fit path, both evaluated on the same fixed
    _RANK_PROBE_TERMS probe grid prony_rank_limits uses (same _prony_reduce
    cache entry, so no new pass over the rows):

      * smoothness > 0 — MacKay's effective number of well-determined
        parameters, gamma = npen - lam * tr(L.T L Sigma) with Sigma the
        posterior covariance, evaluated for the smoothed fit AS IT WOULD BE
        on the probe grid. The fit's own gamma (quality.effective_terms) is
        bounded by its term count and so can never say "the data supports
        more terms than you asked for"; this one can. The exact value would
        take a Newton solve on the probe grid, rejected for cost (1.3 s on a
        broadband file at weak smoothing) and for exposing scipy 1.10.1's
        failure modes at 108 columns. Instead the log-parameterized problem
        is LINEARIZED at the current fit:

          - the fitted spectrum is re-sampled on the probe grid: log E_i
            interpolated log-linearly in log tau, then scaled by
            h_probe / h_fit so the coefficient per node tracks the node
            spacing and the summed modulus is preserved — giving c';
          - the Jacobian of the log-parameterized model there is
            J = R' diag(c'), so the Gauss-Newton Hessian is
            diag(c') R'.T R' diag(c'), plus lam' L.T L with lam' the SAME
            _scaled_smoothness normalization the fit uses, at the probe's
            N and dof;
          - gamma = N_probe - lam' * tr(L.T L inv(H)).

        The linearization assumes the dense optimum has the same spectrum
        SHAPE as the current fit (the per-node coefficient scale is what
        sets each column's leverage in log space) and drops the residual
        curvature diag(r.T J) (tried; it made the estimate meaningless).
        Measured 2026-09-04 against converged 108-term fits of the bundled
        master curves: within ~20% at ANY fit size down to N = 4, within 0.5
        at smoothness 0.04-0.4, ~10-15% low at 0.004 (where the dense fit
        re-converges to exploit its extra terms). The equilibrium column is
        kept with the fitted E_eq when it is interior and dropped when the
        fit clamped it (or solid is False), matching the parameterization
        the fit converged on.

      * smoothness == 0 — the NNLS active-set size on the probe grid: the
        number of nodes non-negative least squares chooses to carry when
        offered a dense grid, which is the lambda = 0 analogue of gamma
        (each active coefficient is a free parameter the data placed; the
        rest are exact zeros). Measured 2.2-2.7 per decade on the bundled
        files at 1% relative error, within ~10% of the default grid's own
        active set. prony_rank_limits' noise_prony is deliberately NOT used
        here: it is the ceiling under the weakest prior and sits 2-4x above
        what either path actually resolves.

    Parameters:
        omega (numpy.ndarray): 1-D array of angular frequencies.
        E_stor (numpy.ndarray): 1-D array of storage-modulus values.
        E_loss (numpy.ndarray): 1-D array of loss-modulus values.
        E_stor_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_stor.
        E_loss_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_loss.
        tau_i (numpy.ndarray): The fit's relaxation grid (extension nodes
            included), ascending — smooth_prony_fit's first output.
        E_i (numpy.ndarray): The fitted coefficients, equilibrium term first
            when solid (len(tau_i) + 1 entries) — smooth_prony_fit's second.
        smoothness (float): The user-facing knob the fit ran with.
        solid (bool): Whether the fit carried an equilibrium term.
        std_scale (float): Uniform positive multiplier on both std arrays,
            exactly as passed to smooth_prony_fit.

    Returns:
        float or None: The resolution in relaxation terms; None when a
        smoothed fit has fewer than two grid nodes, its decaying coefficients
        are not all finite and positive, or the probe Hessian is not
        positive definite.
    """
    tau_probe = _probe_grid(omega)
    R, z = _prony_reduce(
        omega, E_stor, E_loss, E_stor_std, E_loss_std,
        tau_probe, solid, std_scale,
    )
    n_probe = len(tau_probe)
    if not smoothness:
        active, _ = nnls(R, z)
        return float(np.count_nonzero(active[solid:] > 0))

    if len(tau_i) < 2:
        return None
    E_dec = E_i[len(E_i) - len(tau_i):]
    if not (np.all(np.isfinite(E_dec)) and np.all(E_dec > 0)):
        return None
    m_probe = n_probe + solid
    # The residual row is zero across the basis columns; drop it so the Gram
    # below is the plain (m_probe, m_probe) product.
    R = R[:m_probe]
    log_tau_fit, log_tau_probe = np.log(tau_i), np.log(tau_probe)
    h_fit = (log_tau_fit[-1] - log_tau_fit[0]) / (len(tau_i) - 1)
    h_probe = (log_tau_probe[-1] - log_tau_probe[0]) / (n_probe - 1)
    # The probe spans exactly the data window, which the (possibly extended)
    # fit grid contains, so np.interp's edge clamp only absorbs fp rounding.
    c = np.exp(np.interp(log_tau_probe, log_tau_fit, np.log(E_dec))
               + np.log(h_probe / h_fit))
    has_eq = bool(solid and len(E_i) == len(tau_i) + 1 and E_i[0] > 0)
    if has_eq:
        c = np.concatenate(([E_i[0]], c))
    elif solid:
        R = R[:, 1:]
    log_range = log_tau_probe[-1] - log_tau_probe[0]
    lam = _scaled_smoothness(
        smoothness, n_probe, 2 * len(omega) - m_probe, log_range) ** 2
    # Gauss-Newton block of the log-parameterized Hessian, J = R diag(c),
    # then the penalty on its bands. R is a fresh scaled copy (see
    # _prony_reduce), never the read-only cache entry.
    H = (R.T @ R) * c * c[:, None]
    _add_penalty_inplace(H, lam, has_eq)
    try:
        chol = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(chol)):
        return None
    sigma = cho_solve((chol, True), np.eye(len(c)))
    gamma = n_probe - lam * _penalty_trace(sigma, has_eq)
    return float(gamma) if np.isfinite(gamma) else None
