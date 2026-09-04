"""
The smoothed Prony fit: the one entry point that turns measured complex-modulus
data into (tau_i, E_i).

Composes the rest of the fitting stack — `prony` for the grid, `reduction` for
the exact QR compression, `objective` for the penalized loss, `quality` for the
score — and owns the two decisions that need all four in view: how the
user-facing smoothness knob is normalized, and how the equilibrium modulus is
kept out of the Newton solver's search (`_PlateauProjectedProblem`).
"""

import contextlib
import ctypes
import threading

import numpy as np
from scipy.optimize import minimize, nnls

from .prony import prony_relaxation_space
from .objective import _PronyLoss, _scaled_smoothness
from .reduction import _prony_reduce
from .quality import _FitQuality, _prony_fit_quality


# Decades of relaxation-time grid added past each end of the data window on
# the smoothed path, so the fitted spectrum (and its uncertainty band) visibly
# extrapolates instead of stopping dead at 1/omega_max and 1/omega_min. Purely
# aesthetic — the value is a display-tuning knob, hardcoded after a screenshot
# review, NOT a user-facing setting. The extension terms sit outside the data's
# span, so the data cannot determine them: the smoothness penalty does (they
# come out log-linear, level and slope inherited from the window edge via the
# junction curvature), which is exactly the honesty the widening sigma band
# reports. See _extended_relaxation_space for the sizing rules. Chosen from a
# 1/2/3-decade screenshot review of the bundled files (user pick, 2026-08-27).
_GRID_EXTENSION_DECADES = 1.0


# Wall-clock budget for the Newton solve, in seconds. Legitimate solves run
# on the reduced system — row-count independent, ~25 evaluations — and finish
# in milliseconds across the whole 324-case benchmark grid, so 3 s is two to
# three orders of magnitude of headroom, while surfacing a runaway solve as
# this module's actionable 400 well before the gunicorn worker's 60 s
# request timeout SIGKILLs it into an opaque 500 (observed 2026-08-25; see
# _newton_watchdog). Known to be approached by LEGITIMATE solves: weak
# smoothing (0.004) with N near the numerical-rank cap on a broadband file
# (agilus) takes ~500 Newton iterations, 2.5-3 s (measured 2026-09-04) — a
# budget trip there is a slow solve, not a hang, and is out of scope here.
_NEWTON_TIME_BUDGET = 3.0


class SmoothPronyFitTimeout(ValueError):
    """The Newton solve exceeded _NEWTON_TIME_BUDGET.

    A ValueError so the routes' existing except-ValueError arm turns it into
    a 400 whose message reaches the user's snackbar verbatim — this is an
    input-driven condition (the requested grid size, against this data) with
    a user-side remedy, not a server fault.
    """


class SmoothPronyFitDiverged(ValueError):
    """scipy raised on a non-finite array inside the Newton solve.

    scipy 1.10.1's trust-exact subproblem can raise
    ValueError("array must not contain infs or NaNs") from its own cho_solve
    when its damping iteration goes non-finite (observed at N=108 with the
    grid extension off, past the route's own cap). A ValueError for the same
    reason SmoothPronyFitTimeout is one: the routes' except-ValueError arm
    turns it into a 400 whose message names a user-side remedy, instead of
    the opaque 500 scipy's own text would become.
    """


class _NewtonBudgetExceeded(BaseException):
    """Injected into the solver thread by _newton_watchdog.

    BaseException, not Exception, so no library except-Exception handler
    between the injection point and smooth_prony_fit can swallow it.
    """


# TODO(python>=3.12 + scipy>=1.18): retire this machinery, keep the guard.
# scipy >= 1.17 caps the subproblem loop (subproblem_maxiter, default 25) and
# the outer trust-region loop was always capped (maxiter defaults to 200*n),
# so minimize cannot hang there — but "bounded" is not "fast": the worst
# legal case at N=100 is 20,000 outer x 25 inner passes, tens of seconds,
# enough for one slider drag through a bad zone to stack up every sync
# worker. So keep the budget, get it the boring way (verified against scipy
# main, 2026-08-25 — result.status is 1 exactly when the outer loop stopped
# on maxiter; status 2, "bad approximation", stays cosmetic and ignored):
#
#     result = minimize(..., method='trust-exact',
#                       options={'maxiter': 500})  # healthy runs: ~25-90
#     if result.status == 1:
#         raise SmoothPronyFitTimeout(...)  # same 400; reword the message
#                                           # from seconds to an iteration
#                                           # budget
#
# then delete _newton_watchdog, _NewtonBudgetExceeded, _NEWTON_TIME_BUDGET
# and the contextlib/ctypes/threading imports, and replace
# TestNewtonWatchdog's mechanism tests with a maxiter-exhaustion test (a
# tiny 'maxiter' against the zero-eigenvalue fixture in
# TestNewtonHessianShift, with _NEWTON_HESSIAN_SHIFT_EPS patched to 0).
# Three things do NOT retire with it: never return the capped result even
# though it is often near-converged (it would silently break the ~1e-13
# reproducibility contract that makes coefficient diffs meaningful); keep
# _NEWTON_HESSIAN_SHIFT_EPS (a capped subproblem on an exactly singular
# Hessian is a degraded step, not a correct one); and leave the max_prony
# cap in reduction.prony_rank_limits alone — it marks where extra grid
# columns become redundant and has nothing to do with liveness. Upstream
# wart to watch when bumping: the capped subproblem can exit with `p`
# unbound if every pass fails factorization (UnboundLocalError, scipy main
# as of 2026-08-25).
@contextlib.contextmanager
def _newton_watchdog(budget: float):
    """
    Raise _NewtonBudgetExceeded in the calling thread if the body runs longer
    than `budget` seconds.

    Why this exists: scipy's trust-exact subproblem solver
    (_trustregion_exact.IterativeSubproblem.solve) is a `while True` whose
    every exit path requires a Moré-Sorensen stop inequality to hold, and
    when the Hessian has an EXACTLY zero eigenvalue those inequalities can be
    unsatisfiable in float64 — the lambda iteration then cycles forever
    (`self.niter` is counted but never checked, so minimize's maxiter cannot
    help: the outer loop never gets control back). Observed 2026-08-25 on a
    noise-free synthetic 2-decade file: 88 subproblems solved in
    microseconds, the 89th still spinning at 120 s, with the gradient
    already down 7 decades — the fit was done, the solver just could not
    certify its last step. The zero eigenvalue was traced (2026-09-04) to a
    direction that is BOTH in the smoothness penalty's null space (a
    log-linear ramp of the log-coefficients has zero second difference) AND
    invisible to the data (that file is a single Debye at the long-tau end
    of the window with no mass elsewhere, so the ramp runs the remaining
    coefficients down to ~1e-150): a flat valley the subproblem cannot
    bracket. It reproduced at N = 14, 18, 20 on a 2-decade window whose
    numerical-rank cap is 21 — the cap in reduction.prony_rank_limits never
    was the guard and is not documented as one. _NEWTON_HESSIAN_SHIFT_EPS
    lifts that eigenvalue off zero for the solver and removed every
    reproduced case; this watchdog stays as the wall-clock policy boundary,
    and also catches legitimately slow solves.

    Upstream knows: scipy gh-12513 ("Halting problem in trust-exact
    subproblem", open since 2020) was closed by capping that loop at 25
    passes (subproblem_maxiter, scipy 1.17.0), and scipy 1.18.0 additionally
    fixed the loop reusing a STALE Cholesky factor after re-factorization
    (gh-20244) — a correctness bug present in the 1.10.1 this project is
    pinned to (last release supporting Python 3.8) and a plausible driver of
    the observed cycling. If the stack ever moves past 3.8, scipy >= 1.17
    turns a runaway subproblem into a degraded step instead of a hang and
    this watchdog becomes belt-and-suspenders; it should stay regardless, as
    the wall-clock policy boundary.

    Mechanism: a daemon Timer thread calls PyThreadState_SetAsyncExc on this
    thread's id, which schedules the exception at the next bytecode boundary.
    That interrupts the pathological loop because it is pure Python (a few
    small LAPACK calls per pass, each returning promptly); it could NOT
    interrupt a single long-blocking C call, which is fine here and is why
    this is not a general-purpose timeout. Chosen over signal.alarm because
    SIGALRM only works in the main thread — Flask's threaded dev server and
    any threaded WSGI deployment would silently lose the guard — and over a
    worker process because the reduced problem is cheap to ship but the memo
    objects are not.

    Exit protocol: cancel the timer, then clear any injection that is
    scheduled but not yet delivered (SetAsyncExc with NULL). A delivery that
    races past the clear — the body finishing at essentially exactly the
    budget — still unwinds as _NewtonBudgetExceeded through the `with`
    statement, so the caller's except arm reports a timeout for a fit that
    technically completed; that is the policy boundary behaving as a
    boundary, not a leak into unrelated code.
    """
    tid = threading.get_ident()

    def _expire():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(_NewtonBudgetExceeded))

    timer = threading.Timer(budget, _expire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


# Levenberg-style shift applied to the SOLVER's Hessian only, in units of the
# Hessian array's own machine epsilon (64 * eps: 1.4e-14 for float64). The
# exact Hessian is untouched — quality._prony_fit_quality builds its own
# _PronyLoss and factors that for the covariance and the evidence — so the
# objective, its gradient, the optimum and the reported uncertainty are all
# exactly what they were; only the Newton STEP is computed from H + shift * I.
# Why: see _newton_watchdog — an exactly-zero Hessian eigenvalue (a
# penalty-null, data-invisible direction) sends scipy 1.10.1's Moré-Sorensen
# loop into a cycle. Shifting the diagonal by a rounding-level amount gives
# that direction a representable curvature, so the Cholesky-based path of the
# subproblem applies and the zero-gradient direction simply gets a zero step.
# Measured 2026-09-04 (scale = max|diag H|): 1e-14 and 1e-13 removed every
# reproduced hang (N = 14..43, smoothness 0.04 / 0.4 / 1.0, extension off)
# for 10-20% more Newton iterations, and moved real-file solutions by
# <= 8e-8 in log E (relative chi2 <= 5e-12, i.e. inside the gradient
# tolerance's own basin); 1e-12 doubled the iteration count, 1e-10 and above
# cost 3-50x and drifted visibly. Expressed in eps rather than as a bare
# 1e-14 because scipy factors whatever dtype it is handed
# (IterativeSubproblem picks LAPACK potrf by the array's dtype, no upcast):
# the same margin above a float32 rounding floor is 64 * 1.2e-7. Every path
# today builds the Hessian in float64 (prony_basis allocates float64 and the
# QR reduction keeps it), so that branch is future-proofing, not a tested
# regime. Read at call time so tests can patch it to 0 for an unshifted
# reference.
_NEWTON_HESSIAN_SHIFT_EPS = 64


def _shifted_hessian(hess, eps_multiple: float):
    """
    Wrap a Hessian callable so the solver sees H + eps_multiple * eps * max|diag H| * I.

    eps is the machine epsilon of the array `hess` returns, so the shift is a
    fixed number of ulps of the Hessian's largest diagonal entry whatever the
    precision. `hess` must return a FRESH array per call — both
    _PronyLoss.hess and _PlateauProjectedProblem.hess do — since the shift
    is applied in place. eps_multiple == 0 returns hess's output untouched.

    Parameters:
        hess (callable): logcoefs -> fresh (m, m) Hessian array.
        eps_multiple (float): Shift in units of the array's machine epsilon.

    Returns:
        callable: logcoefs -> the shifted Hessian, for minimize's hess=.
    """
    def shifted(logcoefs: np.ndarray) -> np.ndarray:
        H = hess(logcoefs)
        if eps_multiple:
            shift = (eps_multiple * np.finfo(H.dtype).eps
                     * np.abs(np.diag(H)).max())
            np.einsum('ii->i', H)[...] += shift
        return H
    return shifted


def _extended_relaxation_space(
        tau_min: float,
        tau_max: float,
        N: int,
        n_res: int,
        solid: bool,
        extension_decades: float,
) -> tuple:
    """
    The fit grid widened by ~extension_decades per side at the in-window density.

    Adds nodes BEYOND the window rather than stretching the user's N over a
    wider span: the in-window node positions coincide with the unextended
    prony_relaxation_space grid (to fp rounding — nothing may compare grids
    bitwise), so extension changes what the fit reports outside the data, not
    how it resolves inside it. The per-side count is
    round(extension_decades / h_dec) with h_dec the in-window log10 spacing,
    then capped so the whole fit keeps at least one degree of freedom:
    headroom = n_res - 1 - solid - N is how many terms can still be added, and
    each extension step costs two (one per side), so the cap is headroom // 2.
    No extension at all when N < 2 (no spacing to inherit), when the span is
    degenerate, or when there is no headroom — the caller then runs on exactly
    the classic grid.

    The cost of extending is carried where it is visible: dof drops by
    2 * n_ext (shifting the sqrt(dof) in the penalty normalization), and the
    penalty smooths the tail nodes along with the window. Both effects are a
    few percent at ordinary sizes and are accepted — the alternative, a
    separate penalty convention for tail terms, is exactly the kind of drift
    the objective/quality split exists to prevent.

    Parameters:
        tau_min (float): Shortest in-window relaxation time (1 / max(omega)).
        tau_max (float): Longest in-window relaxation time (1 / min(omega)).
        N (int): User-requested number of in-window terms.
        n_res (int): Residual count of the full problem (2 * len(omega)).
        solid (bool): Whether the fit carries an equilibrium term.
        extension_decades (float): Target widening per side, in decades.

    Returns:
        tuple: (tau_i, n_ext) — the grid (length N + 2 * n_ext) and the
        per-side extension count actually applied.
    """
    log_min, log_max = np.log10(tau_min), np.log10(tau_max)
    span_dec = log_max - log_min
    if N < 2 or not (np.isfinite(span_dec) and span_dec > 0):
        return prony_relaxation_space(tau_min, tau_max, N), 0
    h_dec = span_dec / (N - 1)
    headroom = n_res - 1 - solid - N
    n_ext = min(int(round(extension_decades / h_dec)), max(0, headroom // 2))
    if n_ext <= 0:
        return prony_relaxation_space(tau_min, tau_max, N), 0
    tau_i = np.logspace(
        log_min - n_ext * h_dec, log_max + n_ext * h_dec, N + 2 * n_ext,
    )
    return tau_i, n_ext


class _PlateauProjectedProblem:
    """
    The smoothed fit with the equilibrium (plateau) modulus projected out, so
    the solver only searches the log-coefficients of the decaying terms.

    Variable projection. For any fixed decaying coefficients c the optimal
    non-negative equilibrium modulus is a one-variable NNLS with the closed
    form

        E_eq = max(0, r0 . (z - Rr c) / (r0 . r0))

    (r0 the equilibrium column of the reduced basis, Rr the rest), so the
    equilibrium term never has to be a search variable. While E_eq > 0 the
    partially-minimized loss is EXACTLY the solid=False loss on the system
    projected orthogonally to r0, (P z, P Rr); once it clamps at zero it is the
    solid=False loss on (z, Rr) itself. A _PronyLoss on whichever pair applies
    therefore gives the exact loss, gradient and Hessian of the projected
    problem — no envelope-theorem approximation — and shares their common
    intermediates between the solver's separate fun, jac and hess calls.

    Why bother: in log-space the equilibrium term is unpenalized and its
    gradient carries a factor of E_eq itself (chain rule), so a solver can run
    it toward -inf, watch the gradient vanish, and declare convergence at a
    point that is not a minimum of anything. Measured on the bundled VeroCyan
    master curve: a 14x worse objective than the true optimum. The correct
    answer is sometimes E_eq == 0 — that is a boundary optimum, not something
    a penalty should push away from — and log-parameterization turns that
    boundary into a spurious stationary point at infinity. Projection removes
    the direction from the search entirely, guarantees E_eq is optimal for the
    returned decaying terms, and lands the clamped case on an exact 0.0.

    Exposes fun / jac / hess with the same signatures as _PronyLoss (which
    also carries the overflow guard, via log_cap), plus equilibrium() and
    coefficients() to rebuild the full vector. The solid=False fit needs none
    of this and uses a _PronyLoss directly.
    """

    def __init__(self, data: np.ndarray, basis: np.ndarray, smoothness: float,
                 log_cap: float = None):
        r0 = basis[:, 0]
        self._r0 = r0
        self._r0_sq = r0 @ r0
        # Two views of the same problem: the clamped system (z, Rr) — whose
        # residual also yields E_eq — and the system projected orthogonally
        # off r0, for E_eq > 0. The projector is applied once to each array
        # rather than materialized.
        rest = basis[:, 1:]
        self._clamped = _PronyLoss(data, rest, smoothness, False, log_cap)
        self._free = _PronyLoss(
            data - r0 * ((r0 @ data) / self._r0_sq),
            rest - np.outer(r0, (r0 @ rest) / self._r0_sq),
            smoothness, False, log_cap,
        )

    def equilibrium(self, logcoefs: np.ndarray) -> float:
        """Optimal non-negative equilibrium modulus for these decaying terms.

        Exactly 0.0 when the unconstrained optimum is negative — the clamp is
        the active set of a one-variable NNLS, not a rounding artifact.
        """
        resid = self._clamped.residual(logcoefs)
        return max(0.0, (self._r0 @ resid) / self._r0_sq)

    def _loss(self, logcoefs: np.ndarray) -> _PronyLoss:
        return self._free if self.equilibrium(logcoefs) > 0 else self._clamped

    def fun(self, logcoefs: np.ndarray) -> float:
        """Loss, for scipy.optimize.minimize's fun=."""
        return self._loss(logcoefs).fun(logcoefs)

    def jac(self, logcoefs: np.ndarray) -> np.ndarray:
        """Gradient, for scipy.optimize.minimize's jac=."""
        return self._loss(logcoefs).jac(logcoefs)

    def hess(self, logcoefs: np.ndarray) -> np.ndarray:
        """Exact Hessian, for scipy.optimize.minimize's hess=."""
        return self._loss(logcoefs).hess(logcoefs)

    def coefficients(self, logcoefs: np.ndarray) -> np.ndarray:
        """Full coefficient vector, equilibrium term first."""
        return np.concatenate(([self.equilibrium(logcoefs)], np.exp(logcoefs)))


def smooth_prony_fit(
        omega: np.ndarray,
        E_stor: np.ndarray,
        E_loss: np.ndarray,
        E_stor_std: np.ndarray,
        E_loss_std: np.ndarray,
        N: int,
        smoothness: float,
        solid: bool = True,
        return_fit_quality: bool = False,
        std_scale: float = 1.0,
        grid_extension_decades: float = None,
) -> tuple:
    """
    Fit a Prony series to complex-modulus data with coefficient smoothing.

    Builds a log-spaced relaxation-time grid spanning 1/max(omega) to
    1/min(omega) and solves for non-negative Prony coefficients that minimize
    the weighted squared residuals, plus an optional second-difference
    smoothness penalty on the log-coefficients (see _prony_objective).

    Numerics: the weighted data term is first compressed EXACTLY by a chunked
    QR factorization of the weighted basis (see _prony_reduce) —
        ||(y - B c) / std||^2 = ||R c - z||^2
    — so the reduced system has at most N + solid + 1 rows regardless of how many
    data rows the upload carries. Householder QR accumulates the residual
    information backward-stably (no explicit sums of squares), memory stays
    O(_QR_CHUNK_ROWS * N), and every subsequent solver operation costs O(N^2)
    independent of the input row count.

    With smoothness == 0 the reduced problem is exactly non-negative least
    squares and is solved directly by scipy.optimize.nnls: finite,
    deterministic, no line search, no initial guess. Coefficients may then be
    EXACTLY zero (downstream consumers already filter E_i != 0). With
    smoothness > 0 the log-space penalty is nonlinear in the coefficients, so
    the reduced problem is minimized by exact Newton in a trust region
    (scipy's trust-exact, fed _PronyLoss.fun / .jac / .hess) over the
    log-coefficients of the decaying terms, with the equilibrium modulus
    projected out in closed form — see _PlateauProjectedProblem. The seed is
    flat:
    every term at log(max(E_stor) / m).

    Why Newton, and why that seed. The penalty makes the Hessian's condition
    number ~1e12 at ordinary settings, and first-order and limited-memory
    methods cannot follow it: on a 324-case benchmark (four bundled master
    curves plus synthetic 12-40 decade ones, N from 20 to 150, smoothness
    1e-3 to 100) the previous L-BFGS-B solver stopped on its relative-f test
    with the gradient still O(1) in 70 cases — 10x to 100x above the optimum
    on every bundled file at N=100, smoothness=1 — and BFGS, trust-ncg and
    Newton-CG each stalled or overflowed somewhere. trust-exact from the flat
    seed matched the best objective found by any method in all 324 cases, in
    ~25 evaluations (vs ~3800), 10x faster overall. The NNLS solution is NOT
    used as the seed any more: its exact zeros clip into -7 log-unit spikes
    which the penalty turns into an enormous initial gradient, and from there
    trust-exact's subproblem solver was observed to loop without bound
    (an unbounded `while True` in scipy that maxiter cannot cap) while BFGS
    overflowed to NaN. Do not reintroduce it.

    The flat seed does not make that loop unreachable: when the data has no
    mass over part of the grid and the penalty's null space (a log-linear
    ramp) can run those coefficients toward -inf at zero cost, the converged
    Hessian carries an exactly-zero eigenvalue and the subproblem's stop
    inequalities become unsatisfiable in float64 (observed 2026-08-25 on a
    noise-free single-Debye file, hung past 120 s with the gradient already
    down 7 decades; reproduced 2026-09-04 well below the numerical-rank cap,
    which is not a guard against it). Two layers handle that. The Hessian
    handed to scipy is shifted by _NEWTON_HESSIAN_SHIFT_EPS ulps of its
    largest diagonal entry (see _shifted_hessian) — a Levenberg-style
    numerical regularization of the STEP only, never of the objective, the
    gradient, or the scored Hessian — which removed every reproduced hang at
    a cost of 10-20% more iterations and <= 8e-8 in log E. And the solve
    runs under _newton_watchdog, which converts a run past
    _NEWTON_TIME_BUDGET into SmoothPronyFitTimeout — a ValueError, so the
    routes answer 400 with the actionable message instead of the gunicorn
    worker being killed into an opaque 500. A ValueError raised from inside
    scipy's own machinery (its 1.10.1 subproblem can go non-finite) is
    re-raised the same way as SmoothPronyFitDiverged.

    Parameters:
        omega (numpy.ndarray): 1-D array of angular frequencies.
        E_stor (numpy.ndarray): 1-D array of storage-modulus values, same
            length as omega.
        E_loss (numpy.ndarray): 1-D array of loss-modulus values, same length
            as omega.
        E_stor_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_stor, same length as omega. Used to weight residuals.
        E_loss_std (numpy.ndarray): 1-D array of per-point standard deviations
            for E_loss, same length as omega. Used to weight residuals.
        N (int): Number of relaxation times in the fit grid.
        smoothness (float): Strength of the smoothing prior on the
            log-coefficients. Pass 0 to disable. Normalized internally by
            sqrt(dof / h**3) (see _scaled_smoothness), h being the log-tau grid
            spacing, which makes it the exchange rate between the two numbers
            the fit-quality readout reports: V/dof = chi2_reduced +
            smoothness**2 * (log_range * curvature). A given value therefore
            produces comparable smoothing whether the file has 400 rows or
            40,000, whether it is fit with 20 terms or 100, and whether it
            covers 4 decades or 20.
        solid (bool): Whether to include an equilibrium-modulus term.
        return_fit_quality (bool): Append a _FitQuality to the return tuple.
            Off by default so existing two-value unpacking keeps working.
        std_scale (float): Uniform positive multiplier on both std arrays,
            equivalent to passing E_stor_std * std_scale but held out of the
            reduction so a caller that varies ONLY this factor — the
            relative-error widget — reuses one cached reduction across every
            value instead of redoing the O(rows) QR per move. See _prony_reduce.
        grid_extension_decades (float): Test seam for the smoothed path's grid
            extension: None (the default) uses _GRID_EXTENSION_DECADES, 0.0
            disables extension entirely. Production callers never pass it —
            the widening is an internal display decision, not a fit setting.

    Returns:
        tuple: (tau_i, E_i) where tau_i is the 1-D relaxation-time grid and
        E_i is the 1-D non-negative coefficient array of length
        len(tau_i) + bool(solid). On the unsmoothed path tau_i has exactly N
        entries spanning 1/max(omega)..1/min(omega); on the smoothed path it
        additionally carries ~_GRID_EXTENSION_DECADES decades of extension
        nodes per side at the in-window density (see
        _extended_relaxation_space), so its length is N + 2 * n_ext.
        Entries can be exactly zero (NNLS active set). With
        return_fit_quality, (tau_i, E_i, quality); quality.neg_log_posterior is
        None unless the fit converged to an INTERIOR minimum with smoothing on,
        since the Laplace approximation behind it assumes a stationary point
        (when the projected equilibrium modulus clamps at exactly zero the
        score is that of the N-term solid=False problem the solver actually
        converged on), and quality.curvature is None on the unsmoothed path,
        where the NNLS active set makes log-coefficients (and so their
        roughness) undefined. quality.covariance follows the same availability
        as the Laplace machinery — the posterior covariance of the fitted
        log-coefficients, None on the unsmoothed path or when the Hessian is
        not positive definite; on the clamped-equilibrium path it covers the
        N decaying log-coefficients only (no equilibrium row), matching the
        problem the solver converged on. quality.effective_terms is MacKay's
        effective number of well-determined decaying terms,
        npen - lam * tr(L.T L Sigma), available exactly when covariance is;
        being the fit's own count it never exceeds the grid it ran on (see
        reduction.prony_resolution for the dense-grid version).
    """
    assert isinstance(omega, np.ndarray) and omega.ndim == 1, \
        "omega must be a 1-D numpy.ndarray"
    assert isinstance(E_stor, np.ndarray) and E_stor.ndim == 1, \
        "E_stor must be a 1-D numpy.ndarray"
    assert isinstance(E_loss, np.ndarray) and E_loss.ndim == 1, \
        "E_loss must be a 1-D numpy.ndarray"
    assert len(omega) == len(E_stor) == len(E_loss), \
        "omega, E_stor, E_loss must all have the same length"
    assert isinstance(E_stor_std, np.ndarray) and E_stor_std.shape == E_stor.shape, \
        "E_stor_std must be a 1-D numpy.ndarray matching E_stor"
    assert isinstance(E_loss_std, np.ndarray) and E_loss_std.shape == E_loss.shape, \
        "E_loss_std must be a 1-D numpy.ndarray matching E_loss"
    # Not merely "non-zero": a negative scale would flip the sign of every
    # weighted residual, and the route already rejects relative_error <= 0
    # and error_scale <= 0 (whichever feeds std_scale).
    assert std_scale > 0, "std_scale must be positive"

    tau_max = 1 / np.min(omega)
    tau_min = 1 / np.max(omega)
    n_res = 2 * len(omega)

    # Reduced problem with smoothness == 0 is exactly non-negative least
    # squares — solve it directly (finite algorithm, no iteration budget) on
    # the EXACT unextended grid. The grid extension below is deliberately not
    # applied here: with no penalty the tail terms would be pure numerical
    # null space (nothing determines them, and the active set is not
    # guaranteed to zero them), and no covariance exists to widen a band over
    # the extrapolation. The visible consequence is an x-range jump when the
    # smoothness slider crosses zero; that is honest, not a glitch.
    if smoothness == 0:
        tau_i = prony_relaxation_space(tau_min, tau_max, N)
        m = N + solid
        dof = n_res - m
        R, z = _prony_reduce(
            omega, E_stor, E_loss, E_stor_std, E_loss_std, tau_i, solid,
            std_scale,
        )
        E_nnls, rnorm = nnls(R, z)
        if not return_fit_quality:
            return tau_i, E_nnls
        # No penalty means no posterior over lam to report, but the misfit is
        # still meaningful — and nnls already handed us ||R c - z||, which the
        # reduction's residual row makes a full-problem quantity. Curvature is
        # genuinely undefined here, not merely unavailable: NNLS's active set
        # leaves coefficients EXACTLY zero, whose logs are -inf.
        return tau_i, E_nnls, _FitQuality(
            rnorm ** 2 / dof if dof > 0 else None, None, None,
        )

    # smoothness > 0: widen the grid a couple of decades past the data window
    # (see _GRID_EXTENSION_DECADES) so the reported spectrum extrapolates,
    # then run Newton on the reduced system with the equilibrium term
    # projected out. Everything downstream — m, dof, the penalty
    # normalization, the flat seed, the quality score — reads the EXTENDED
    # quantities, so the extension terms are ordinary fit terms in every
    # respect except that only the penalty determines them.
    if grid_extension_decades is None:
        grid_extension_decades = _GRID_EXTENSION_DECADES
    tau_i, n_ext = _extended_relaxation_space(
        tau_min, tau_max, N, n_res, solid, grid_extension_decades,
    )
    N_total = len(tau_i)
    m = N_total + solid
    dof = n_res - m

    R, z = _prony_reduce(
        omega, E_stor, E_loss, E_stor_std, E_loss_std, tau_i, solid, std_scale
    )
    # (R, z) carries the unreachable orthogonal residual as its last row, which
    # is what puts every score on the full-problem scale. The Newton solver
    # would be indifferent — a constant row changes neither gradient nor
    # Hessian, and trust-exact stops on the gradient, not on a relative
    # reduction of f the way L-BFGS-B did — so the slice is only about not
    # carrying a dead row through every evaluation.
    R_fit, z_fit = R[:m], z[:m]

    # Normalize the knob so it means the same thing on any upload; see
    # _scaled_smoothness, which _prony_fit_quality re-derives from the same
    # inputs so the reported score belongs to the fit that was actually run.
    log_range = np.log(tau_i[-1] / tau_i[0])
    smoothness_scaled = _scaled_smoothness(smoothness, N_total, dof, log_range)
    # No single Prony term above ~1000x the data maximum: the overflow guard
    # in _PronyLoss, same physical cap the old L-BFGS-B upper bound encoded.
    log_cap = np.log(E_stor.max()) + np.log(1e3)
    if solid:
        problem = _PlateauProjectedProblem(
            z_fit, R_fit, smoothness_scaled, log_cap)
    else:
        problem = _PronyLoss(z_fit, R_fit, smoothness_scaled, False, log_cap)
    # Flat seed, data-scaled: zero curvature, so the penalty contributes
    # nothing to the first step however large its weight.
    x0 = np.full(N_total, np.log(E_stor.max() / m))
    try:
        with _newton_watchdog(_NEWTON_TIME_BUDGET), \
                np.errstate(over='ignore', invalid='ignore'):
            result = minimize(
                fun=problem.fun,
                x0=x0,
                jac=problem.jac,
                hess=_shifted_hessian(problem.hess, _NEWTON_HESSIAN_SHIFT_EPS),
                method='trust-exact',
            )
    except _NewtonBudgetExceeded:
        raise SmoothPronyFitTimeout(
            f"The fit did not converge within {_NEWTON_TIME_BUDGET:.0f} "
            f"seconds at a relaxation grid size of {N} — lower the "
            f"relaxation grid size, or raise the smoothness or the assumed "
            f"error."
        ) from None
    except ValueError as exc:
        # scipy 1.10.1's subproblem can raise "array must not contain infs
        # or NaNs" from its own cho_solve when its damping iteration goes
        # non-finite (observed at N=108 with the grid extension off). Same
        # 400 path as the timeout, with a message that names a remedy.
        raise SmoothPronyFitDiverged(
            f"The fit broke down (non-finite values inside the solver) at a "
            f"relaxation grid size of {N} — lower the relaxation grid size, "
            f"or raise the smoothness or the assumed error."
        ) from exc
    # result.success is deliberately not consulted: near the optimum the
    # trust radius can collapse on a precision-limited reduction ratio and
    # scipy reports "bad approximation" with the gradient already ~1e-5.
    E_i = problem.coefficients(result.x) if solid else np.exp(result.x)
    if not return_fit_quality:
        return tau_i, E_i

    if solid and E_i[0] == 0:
        # The projected equilibrium modulus clamped at zero, so in the full
        # parameterization the optimum sits on a coefficient boundary:
        # log(E_eq) = -inf and the Laplace expansion has no curvature in that
        # direction. What the solver actually converged on there is the
        # solid=False problem in the N decaying terms (see
        # _PlateauProjectedProblem),
        # and that problem's interior minimum IS this point — so score it as
        # that: the posterior given the active set, the same convention NNLS
        # uses for its exact zeros. n_resid is lowered by one so the dof
        # _prony_fit_quality derives — and hence the penalty weight it rebuilds
        # — stay exactly the fit's own: the pinned equilibrium term is still
        # one of the fit's m parameters.
        quality = _prony_fit_quality(
            result.x, z, R[:, 1:], smoothness, False,
            n_resid=n_res - 1,
            log_range=log_range,
        )
        return tau_i, E_i, quality

    quality = _prony_fit_quality(
        np.log(E_i), z, R, smoothness, solid,
        n_resid=n_res,
        log_range=log_range,
    )
    return tau_i, E_i, quality
