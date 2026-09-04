"""
Everything that turns fit results into the payload the browser renders: the
plotly figures, the captions stamped on them, the thinning that keeps a
41k-row upload from bloating the response, and the coefficient table that
accompanies the figures.

Presentation only — nothing here changes a number the fit produced. `chart`
orchestrates the calls; the figures go out as JSON via the route.
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .prony import compute_complex, compute_relaxation_modulus
from .shift import hybrid_shift, wlf_log10_shift
from .tts import MAX_ABS_LOG10_SHIFT
from .uncertainty import (
    _SIGMA_DISPLAY_CAP,
    _split_terms,
    sigma_log_coefficients,
    spectrum_error_bars,
    complex_modulus_noise,
    complex_modulus_sigma,
    relaxation_sigma,
)


# Experiment traces with more rows than this are thinned before plotting —
# broadband uploads (e.g. 41k-row chirp master curves) otherwise bloat the
# response JSON and bog down browser-side plotly rendering. This affects the
# FIGURES ONLY: the Prony fit and the coefficient table always use every row.
# ~2000 points per trace is far denser than any screen resolves.
_PLOT_MAX_POINTS = 2000

# Figure notices — the decimation warning and the fit-quality readout — are gray
# right-aligned captions in paper coordinates above the plot, one per row. They
# cannot share a row: each runs 85-100 characters, so even anchored to opposite
# edges they collided in the middle at every width the frontend renders at. Row
# 0 sits at _NOTICE_BASE_Y nearest the plot; every row above it needs
# _NOTICE_ROW_MARGIN more headroom than the _NOTICE_TOP_MARGIN plotly express
# leaves on these faceted figures.
_NOTICE_BASE_Y = 1.06
_NOTICE_ROW_STEP = 0.065
_NOTICE_TOP_MARGIN = 60
_NOTICE_ROW_MARGIN = 22
_NOTICE_NAME = 'figure-notice'


def _decimate_for_plot(df: pd.DataFrame) -> tuple:
    """
    Thin a sorted experiment DataFrame for plotting when it exceeds
    _PLOT_MAX_POINTS.

    Rows are subsampled at evenly spaced positional indices (first and last
    rows always kept), which preserves the curve shape for data that is
    already sorted along its x axis regardless of grid spacing. The fit never
    sees this — callers decimate only the frames handed to figure builders.

    Parameters:
        df (pd.DataFrame): Experiment data sorted by its x column.

    Returns:
        tuple: (plot_df, percent) where plot_df is df itself when no thinning
        was needed, or a positional subsample otherwise; percent is None when
        no thinning happened, else the integer percentage of rows dropped
        (for the user-facing figure annotation).
    """
    n = len(df)
    if n <= _PLOT_MAX_POINTS:
        return df, None
    idx = np.unique(np.linspace(0, n - 1, _PLOT_MAX_POINTS).astype(int))
    percent = int(round(100.0 * (1 - len(idx) / n)))
    return df.iloc[idx], percent


def _stamp_notice(figs, text: str) -> None:
    """
    Caption each figure above the plot area, on the next free row.

    Notices stack upward in call order: each counts the notices already on the
    figure and takes the row above them, so two long captions never share a
    line while a solitary caption still sits on the bottom row. The top margin
    grows to match, since a stacked row would otherwise be clipped.

    All captions are right-aligned — ragged left, flush right. The fit-quality
    readout's numbers change width from one fit to the next, and anchoring the
    right edge keeps that from shifting the whole block sideways as a user drags
    the sliders.

    The captions ride inside the figures, so to_json carries them to the browser
    with no frontend work. The MARGIN does need help getting there: PlotlyView
    imposes its own layout on every chart, so it merges the server's margin over
    its defaults specifically to let the headroom below survive the trip.

    Each caption is tagged with a name so the row count sees only these: plotly
    express has already put the facet titles in layout.annotations.

    Parameters:
        figs: Iterable of plotly Figures to caption.
        text (str): Caption text.
    """
    for fig in figs:
        row = sum(1 for a in fig.layout.annotations if a.name == _NOTICE_NAME)
        fig.add_annotation(
            name=_NOTICE_NAME, text=text,
            xref='paper', yref='paper',
            x=1.0, y=_NOTICE_BASE_Y + row * _NOTICE_ROW_STEP,
            xanchor='right', yanchor='bottom', showarrow=False,
            font=dict(size=11, color='gray'),
        )
        fig.update_layout(margin_t=max(
            fig.layout.margin.t or _NOTICE_TOP_MARGIN,
            _NOTICE_TOP_MARGIN + row * _NOTICE_ROW_MARGIN,
        ))


def _annotate_decimation(figs, percent) -> None:
    """
    Stamp a decimation notice onto each figure when plot thinning occurred.

    No-op when percent is None. Stamped after the fit-quality readout, so on
    the figures that carry both this one takes the upper row: it says the same
    thing on every slider move, where the readout is the number being watched.

    Parameters:
        figs: Iterable of plotly Figures to annotate.
        percent: Integer percentage of experiment rows dropped, or None.
    """
    if percent is None:
        return
    _stamp_notice(figs, (
        f"too many data points, plot traces decimated by {percent}% for speed"
        " (the fit uses all points)"
    ))


def _annotate_grid_suggestion(figs, requested_n, resolution, max_prony) -> None:
    """
    Suggest a finer relaxation grid when the data resolves more terms than
    the request offers.

    resolution is the data's term-count resolution under the SELECTED error
    profile and smoothing, from reduction.prony_resolution: the dense-grid
    effective parameter count gamma (rounded) when smoothing is on, the
    probe-grid NNLS active-set size when it is off. Fires only when
    requested_n < resolution — a grid coarser than what the data can place —
    and suggests min(max_prony, ceil(1.5 * resolution)): 1.5x because gamma
    counts well-determined DIRECTIONS and a grid needs nodes around each to
    place them (the 3-per-decade default sits at ~1.5x the measured
    resolution of the bundled files); capped at max_prony because past the
    numerical rank extra nodes are redundant columns. No stamp when the
    suggestion would not actually raise N (the cap already binds), when
    resolution is None (nothing to measure against), or when the request
    already meets the resolution — the healthy case adds no visual noise,
    and a request ABOVE the resolution is not called out at all: the extra
    terms are then the smoothing's to fill, which is what smoothing is for.

    The raw resolution is deliberately not surfaced — it is an estimate with
    its own ~20% error bar, and the user's action is the grid size, not the
    count — only the scaled, capped suggestion is. Still conditional on the
    error profile ("if the error profile is accurate"): resolution is a
    statement about the user's stated uncertainty, not about the data alone.
    The previous caption here read "at most ~k terms are identifiable; the
    rest is smoothing" from prony_rank_limits' noise count; that count is a
    smoothness-free ceiling 2-4x above what any smoothed fit resolves, so it
    never fired at a sensible N and, when it did, named a number no action
    followed from (retired 2026-09-04).

    Stamped after the fit-quality readout and the decimation notice, so it
    takes the top row: like the decimation notice, it says the same thing on
    every slider move.

    Parameters:
        figs: Iterable of plotly Figures to annotate.
        requested_n (int): The effective Prony term count the fit ran with.
        resolution (int or None): The data's resolution in terms, or None.
        max_prony (int or None): The numerical-rank cap on the grid size.
    """
    if resolution is None or max_prony is None or requested_n >= resolution:
        return
    n_suggest = min(int(max_prony), int(math.ceil(1.5 * resolution)))
    if n_suggest <= requested_n:
        return
    _stamp_notice(figs, (
        f"if the error profile is accurate, the data can support more "
        f"terms; try a relaxation grid size of {n_suggest}"
    ))


def _annotate_fit_quality(figs, quality) -> None:
    """
    Stamp the fit-quality readout onto each figure that overlays fit on data.

    Stamped before the decimation notice so it takes the bottom row, nearest
    the plot — see _stamp_notice. Sharing one row with that notice was not
    enough: both strings are long enough to cross the middle of the plot.

    All three numbers are "lower is better". Fields that are None are omitted,
    so an unsmoothed fit shows the misfit alone (with no smoothing there is
    neither a posterior over the smoothing weight nor a defined roughness).
    The nu in chi2/nu is the EFFECTIVE degrees of freedom on the smoothed
    path (quality.effective_terms), the classical count on the unsmoothed one.
    Curvature sits in the middle, next to chi-squared: those two are the L-curve
    coordinates a user trades off when sweeping the smoothness slider.

    Parameters:
        figs: Iterable of plotly Figures to annotate.
        quality (_FitQuality): Scores from smooth_prony_fit, or None to no-op.
    """
    if quality is None:
        return
    parts = []
    if quality.chi2_reduced is not None:
        parts.append(f"misfit (χ²/ν) = {quality.chi2_reduced:.3g}")
    if quality.curvature is not None:
        parts.append(f"curvature (⟨H″²⟩) = {quality.curvature:.3g}")
    if quality.neg_log_posterior is not None:
        parts.append(
            f"surprisal (−log π(λ)) = {quality.neg_log_posterior:.4g}"
        )
    if not parts:
        return
    _stamp_notice(figs, " | ".join(parts + ["lower is better"]))


# Opacity of the ±1σ ribbons — light enough that the experiment trace stays
# readable through a band drawn UNDER it (see _prepend_traces).
_BAND_ALPHA = 0.25

# Legend names for the two ribbon kinds. Each doubles as the legendgroup, so
# one legend entry toggles every ribbon of its kind at once while the two
# kinds toggle independently. Color follows the trace each band describes:
# the credible band (where the underlying curve lies) takes the Prony line's
# color, the prediction band (where a NEW measurement would land — credible
# variance plus measurement noise) takes the Experiment trace's, both at
# _BAND_ALPHA. Draw order is prediction, credible, data, fit — the prediction
# band is the wider by construction, so each layer stays visible.
_CRED_LEGEND = '±1σ credible'
_PRED_LEGEND = '±1σ prediction'


def _rgba(color: str, alpha: float) -> str:
    """
    An rgba() version of a plotly trace color, for translucent band fills.

    Handles the two forms px actually emits — '#rrggbb' hex from the default
    colorway and 'rgb(...)' strings — and falls back to the color unchanged
    (an opaque band, ugly but correct) for anything else.
    """
    color = color.strip()
    if color.startswith('#') and len(color) == 7:
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        return f'rgba({r},{g},{b},{alpha})'
    if color.startswith('rgb(') and color.endswith(')'):
        return f'rgba({color[4:-1]},{alpha})'
    return color


def _trace_line_color(fig, name_fragment: str, fallback: str) -> str:
    """The line color px gave the first trace whose name contains the
    fragment — read off the BUILT figure rather than hardcoded, so the bands
    keep matching if the colorway or trace order ever changes."""
    for t in fig.data:
        if name_fragment in (t.name or '') and getattr(t.line, 'color', None):
            return t.line.color
    return fallback


def _prony_line_color(fig) -> str:
    """The Prony overlay's line color; the credible band's base color.

    The fallback is px's second default slot, the overlay's usual position
    alongside 'Experiment' — but on single-color figures (fig2) px hands the
    overlay slot 0, and reading the built trace keeps the band matching."""
    return _trace_line_color(fig, 'Term Prony', '#EF553B')


def _experiment_line_color(fig) -> str:
    """The Experiment trace's line color; the prediction band's base color —
    the band describes where new DATA would land, so it wears the data's
    color. Fallback is px's first default slot."""
    return _trace_line_color(fig, 'Experiment', '#636EFA')


def _prepend_traces(fig, traces) -> None:
    """
    Insert traces BEFORE the figure's existing ones, in the given order.

    Band ribbons must draw under both the experiment and fit lines, and
    fill='tonexty' binds each upper band edge to the trace immediately before
    it in DATA order — so each (lower, upper) pair has to land adjacent at the
    front. plotly only accepts NEW traces via add_traces, so append then
    rotate.
    """
    traces = tuple(traces)
    if not traces:
        return
    n = len(traces)
    fig.add_traces(traces)
    fig.data = fig.data[-n:] + fig.data[:-n]


def _band_pair(x, y, sigma, fillcolor: str, log_y: bool,
               xaxis: str = None, yaxis: str = None,
               showlegend: bool = False, name: str = _CRED_LEGEND) -> tuple:
    """
    The (lower, upper) go.Scatter pair of a ±1σ ribbon around curve y.

    The upper edge sits at y + σ with σ capped at y * (e^cap - 1) — the
    six-decade display ceiling (_SIGMA_DISPLAY_CAP) that keeps an
    unconstrained tail from detonating a log axis's autorange or stdlib json.
    On log panels the lower edge is the harmonic form y² / (y + σ_capped):
    positive by construction, exactly log-symmetric with the capped upper edge
    (y·e^s above pairs with y·e^-s below), and equal to y - σ to O((σ/y)²)
    where the band is tight. Linear panels use the plain max(y - σ, 0).

    Both traces are invisible lines (the fill is the band), skip hover so they
    don't shadow the curves, and share `name` as their legendgroup; exactly
    one trace per FIGURE per band kind should pass showlegend=True. On
    faceted figures pass the facet's axis pair ('x'/'y' for column 1,
    'x2'/'y2' for column 2) explicitly.

    Returns:
        tuple: (lower, upper) traces — keep them adjacent, lower first, since
        fill='tonexty' fills from the PREVIOUS trace in data order.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    capped = np.minimum(sigma, y * np.expm1(_SIGMA_DISPLAY_CAP))
    upper = y + capped
    lower = y * y / (y + capped) if log_y else np.maximum(y - sigma, 0.0)
    common = dict(
        mode='lines', line=dict(width=0),
        name=name, legendgroup=name,
        hoverinfo='skip', showlegend=False,
    )
    if xaxis is not None:
        common.update(xaxis=xaxis, yaxis=yaxis)
    return (
        go.Scatter(x=x, y=lower, **common),
        go.Scatter(x=x, y=upper, fill='tonexty', fillcolor=fillcolor,
                   **{**common, 'showlegend': showlegend}),
    )


def _place_tan_delta_axis(fig) -> None:
    """
    Keep the tan delta facet's tick labels clear of both neighbors.

    tan delta is dimensionless and cannot share the modulus panel's scale, so
    its facet needs tick labels of its own. On the default left side they sit
    in the narrow gap between the facets and overlap the plot to their left, so
    they go on the outside edge instead — the only place they fit without
    taking width from the plots.

    That lands them where the legend sits: plotly's automargin reserves room on
    the right for the legend but not for tick labels, so the two are drawn over
    each other. Nudging the legend past its 1.02 default clears the labels, and
    automargin widens the margin to match, so nothing runs off the figure.

    The offset can only be given in paper units — a fraction of the plot width —
    while the labels it has to clear are a fixed pixel width, so the gap closes
    as the figure narrows. 1.10 keeps them apart down to roughly a 500px figure,
    which covers every width PlotlyView asks for on a desktop viewport.

    Parameters:
        fig: Two-facet Figure whose second column is tan delta.
    """
    fig.update_yaxes(side='right', col=2)
    fig.update_layout(legend_x=1.10)


def _build_temperature_figures(temp_sweep_data: pd.DataFrame) -> tuple:
    """
    Build E vs Temperature and tan-delta vs Temperature figures.

    Parameters:
        temp_sweep_data (pd.DataFrame): Frame with columns
            ['Temperature', "E'", "E''"]; extra columns are ignored.

    Returns:
        tuple: (fig4, fig41) where fig4 is the E' / E'' line plot in
        Temperature and fig41 is the E' / tan-delta line plot in Temperature.
    """
    df_melt = pd.melt(
        temp_sweep_data,
        id_vars=["Temperature"],
        value_vars=["E'", "E''"],
        var_name='Modulus',
        value_name="Modulus (Pa)",
    )
    df_melt["Type"] = "Experiment"

    fig4 = px.line(
        df_melt, x="Temperature", y="Modulus (Pa)",
        log_y=True,
        facet_col='Modulus',
        color="Type", line_dash="Type",
        labels={"Temperature": "Temperature (C)"},
    )

    df41_concat = df_melt.copy()
    df41_tand = pd.DataFrame()
    df41_tand["Temperature"] = df41_concat[df41_concat["Modulus"] == "E''"]["Temperature"]
    df41_tand["Type"] = df41_concat[df41_concat["Modulus"] == "E''"]["Type"]
    df41_tand["Modulus (Pa)"] = (
        df41_concat[df41_concat["Modulus"] == "E''"]["Modulus (Pa)"].to_numpy() /
        df41_concat[df41_concat["Modulus"] == "E'"]["Modulus (Pa)"].to_numpy()
    )
    df41_tand['Modulus'] = 'tan delta'
    df41_concat = pd.concat([df41_concat, df41_tand], ignore_index=True)

    fig41 = px.line(
        df41_concat[df41_concat['Modulus'] != "E''"],
        x="Temperature", y="Modulus (Pa)",
        facet_col='Modulus',
        color="Type", line_dash="Type",
        labels={"Temperature": "Temperature (C)"},
    )
    fig41.update_yaxes(matches=None, showticklabels=True)
    fig41.update_yaxes(type="log", col=1)
    _place_tan_delta_axis(fig41)
    fig4.update_yaxes(exponentformat='power')
    fig41.update_yaxes(exponentformat='power')
    return fig4, fig41


def _build_complex_figures(df: pd.DataFrame, tau_i: np.ndarray, E_i: np.ndarray,
                           N_nz: int, covariance: np.ndarray = None,
                           noise: tuple = None) -> tuple:
    """
    Build E vs frequency and tan-delta vs frequency figures with Prony overlay.

    Parameters:
        df (pd.DataFrame): Experimental data with columns
            ['Frequency', 'E Storage', 'E Loss'].
        tau_i (numpy.ndarray): Prony relaxation times.
        E_i (numpy.ndarray): Prony coefficients (length tau_i or tau_i + 1).
        N_nz (int): Number of nonzero DECAYING Prony coefficients, i.e. the
            equilibrium term excluded; used in trace names. Matches the row
            count of the coefficient table _build_coef_records returns.
        covariance (numpy.ndarray): Posterior covariance of the fitted
            log-coefficients (quality.covariance), or None for no ±1σ
            ribbons at all. Bands are evaluated on the SAME frequency grid as
            the overlay curve — compute_complex's own column — so they cannot
            drift onto a different grid.
        noise (tuple): (omega_data, rel_stor, rel_loss) — the measured
            frequencies and the RELATIVE noise profile sigma / |E*| the fit
            ran with (std_scale applied) — or None to skip the prediction
            ribbons. These figures are the only ones that get a prediction
            band: the frequency-domain moduli are what the instrument
            actually measures, so "where would a new reading land" is a real
            question here, where E(t) and the spectrum are interconversions
            with no direct observation to predict.

    Returns:
        tuple: (fig1, fig11) where fig1 is E' / E'' vs Frequency and fig11 is
        E' / tan-delta vs Frequency. With a covariance each carries ±1σ
        credible ribbons under the curves; with noise as well, wider ±1σ
        prediction ribbons under those (credible + measurement noise in
        quadrature). Draw order is prediction, credible, data, fit.
    """
    complex_df = compute_complex(tau_i, E_i)
    x_col, y_col, z_col = df.columns[0], df.columns[1], df.columns[2]
    df_melt = pd.melt(
        df, id_vars=[x_col], value_vars=[y_col, z_col],
        var_name='Modulus', value_name="Modulus (Pa)",
    )
    df_melt["Type"] = "Experiment"

    cx_x, cx_y, cx_z = complex_df.columns[0], complex_df.columns[1], complex_df.columns[2]
    complex_melt = pd.melt(
        complex_df, id_vars=[cx_x], value_vars=[cx_y, cx_z],
        var_name='Modulus', value_name="Modulus (Pa)",
    )
    complex_melt["Type"] = f"{N_nz}-Term Prony"

    df_concat = pd.concat([df_melt, complex_melt], ignore_index=True)

    fig1 = px.line(
        df_concat, x=cx_x, y="Modulus (Pa)",
        log_x=True, log_y=True,
        facet_col='Modulus',
        color="Type", line_dash="Type",
        line_dash_map={"Experiment": "solid", f"{N_nz}-Term Prony": "dash"},
        labels={"Frequency": "Frequency (Hz)"},
    )

    df11_concat = df_concat.copy()
    df11_tand = pd.DataFrame()
    df11_tand["Frequency"] = df11_concat[df11_concat["Modulus"] == "E Loss"]["Frequency"]
    df11_tand["Type"] = df11_concat[df11_concat["Modulus"] == "E Loss"]["Type"]
    df11_tand["Modulus (Pa)"] = (
        df11_concat[df11_concat["Modulus"] == "E Loss"]["Modulus (Pa)"].to_numpy() /
        df11_concat[df11_concat["Modulus"] == "E Storage"]["Modulus (Pa)"].to_numpy()
    )
    df11_tand['Modulus'] = 'tan delta'
    df11_concat = pd.concat([df11_concat, df11_tand], ignore_index=True)

    fig11 = px.line(
        df11_concat[df11_concat['Modulus'] != "E Loss"],
        x="Frequency", y="Modulus (Pa)",
        log_x=True,
        facet_col='Modulus',
        color="Type", line_dash="Type",
        line_dash_map={"Experiment": "solid", f"{N_nz}-Term Prony": "dash"},
        labels={"Frequency": "Frequency (Hz)"},
    )
    fig11.update_yaxes(matches=None, showticklabels=True)
    fig11.update_yaxes(type="log", col=1)
    _place_tan_delta_axis(fig11)

    if covariance is not None:
        freq = complex_df[cx_x].to_numpy()
        sig = complex_modulus_sigma(freq, tau_i, E_i, covariance)
        pred = None
        if noise is not None:
            omega_data, rel_stor, rel_loss = noise
            data_sig = complex_modulus_noise(
                freq, tau_i, E_i, omega_data, rel_stor, rel_loss)
            # Prediction = credible + measurement noise, in quadrature.
            pred = {key: np.hypot(sig[key], data_sig[key]) for key in sig}
        stor = complex_df[cx_y].to_numpy()
        loss = complex_df[cx_z].to_numpy()
        # Facet columns by data order of the melt: col 1 = E Storage on
        # ('x','y'), col 2 = E Loss (fig1) / tan delta (fig11) on ('x2','y2').
        # Prepend order = draw order: the (wider) prediction ribbons paint
        # first, the credible ribbons over them, then the px traces.
        for fig, col2_key, y_col2, col2_log in (
            (fig1, 'E Loss', loss, True),
            (fig11, 'tan delta', loss / stor, False),
        ):
            bands = []
            if pred is not None:
                fill_p = _rgba(_experiment_line_color(fig), _BAND_ALPHA)
                bands += [
                    *_band_pair(freq, stor, pred['E Storage'], fill_p,
                                log_y=True, xaxis='x', yaxis='y',
                                showlegend=True, name=_PRED_LEGEND),
                    *_band_pair(freq, y_col2, pred[col2_key], fill_p,
                                log_y=col2_log, xaxis='x2', yaxis='y2',
                                name=_PRED_LEGEND),
                ]
            fill_c = _rgba(_prony_line_color(fig), _BAND_ALPHA)
            bands += [
                *_band_pair(freq, stor, sig['E Storage'], fill_c, log_y=True,
                            xaxis='x', yaxis='y', showlegend=True,
                            name=_CRED_LEGEND),
                *_band_pair(freq, y_col2, sig[col2_key], fill_c,
                            log_y=col2_log, xaxis='x2', yaxis='y2',
                            name=_CRED_LEGEND),
            ]
            _prepend_traces(fig, bands)

    for fig in (fig1, fig11):
        fig.update_xaxes(exponentformat='power')
        fig.update_yaxes(exponentformat='power')
    return fig1, fig11


def _build_relaxation_figures(tau_i: np.ndarray, E_i: np.ndarray, N_nz: int,
                              fit_settings: bool,
                              covariance: np.ndarray = None) -> tuple:
    """
    Build relaxation-modulus and discrete-spectrum figures.

    Parameters:
        tau_i (numpy.ndarray): Prony relaxation times.
        E_i (numpy.ndarray): Prony coefficients (length tau_i or tau_i + 1).
        N_nz (int): Number of nonzero DECAYING Prony coefficients, i.e. the
            equilibrium term excluded; used in trace names. fig3 splits that
            term into its own long-term-modulus trace, so the same count labels
            both figures.
        fit_settings (bool): If True, overlay the basis scatter on the
            relaxation-modulus figure; if False, return only its line trace.
        covariance (numpy.ndarray): Posterior covariance of the fitted
            log-coefficients, or None for no uncertainty display. Gives fig2 a
            ±1σ ribbon and fig3 asymmetric per-coefficient error bars (the
            Long-Term Modulus line deliberately carries none).

    Returns:
        tuple: (fig2, fig3) where fig2 is the time-domain relaxation modulus
        E(t) and fig3 is the discrete relaxation spectrum — the Prony
        coefficients as dots at (tau_i, E_i) with a horizontal reference line
        at the long-term (equilibrium) modulus when one is present.
    """
    relax = compute_relaxation_modulus(tau_i, E_i)
    relax["Type"] = f"{N_nz}-Term Prony"
    fig2a = px.line(
        relax, x="Time", y="E",
        log_x=True, log_y=True,
        color="Type", line_dash="Type",
        line_dash_map={"Basis": "solid", f"{N_nz}-Term Prony": "dash"},
        labels={"Time": "Time (s)", "E": "Relaxation Modulus (Pa)"},
    )
    fig2a.update_layout(
        autosize=False, width=800, height=450,
        margin=dict(l=80, r=60, t=60, b=80),
    )
    if covariance is not None:
        # Prepended to fig2a BEFORE the overlay merge below, so both
        # fit_settings paths (merged figure and bare fig2a) carry the ribbon.
        t = relax["Time"].to_numpy()
        _prepend_traces(fig2a, _band_pair(
            t, relax["E"].to_numpy(),
            relaxation_sigma(t, tau_i, E_i, covariance),
            _rgba(_prony_line_color(fig2a), _BAND_ALPHA),
            log_y=True, showlegend=True,
        ))

    basis_df = pd.DataFrame({
        "Time": tau_i,
        "E": E_i[len(E_i) - len(tau_i):],
        "Type": f"{N_nz}-Term Basis",
    })
    fig2b = px.scatter(
        basis_df, x="Time", y="E",
        log_x=True, log_y=True,
        symbol="Type",
        labels={"Time": "Time (s)", "E": "Relaxation Modulus (Pa)"},
    )

    fig2 = go.Figure(data=fig2a.data + fig2b.data)
    fig2.update_xaxes(type="log")
    fig2.update_yaxes(type="log")
    fig2.update_layout(
        autosize=False, margin=dict(l=80, r=60, t=60, b=80),
        xaxis_title="Time (s)",
        yaxis_title="Relaxation Modulus (Pa)",
        legend_title="Type",
    )

    # fig3: the discrete relaxation spectrum — the fitted Prony coefficients
    # as dots at (tau_i, E_i) — with the equilibrium term, when present and
    # nonzero, drawn as a horizontal long-term-modulus reference line. Unlike
    # fig2's basis overlay this is the figure's primary content, so
    # fit_settings does not alter it.
    solid = len(E_i) != len(tau_i)
    spectrum_df = pd.DataFrame({
        "Time": tau_i,
        "E": E_i[solid:],
        "Type": f"{N_nz}-Term Prony",
    })
    error_kwargs = {}
    if covariance is not None:
        # Asymmetric error bars from the log-space sigmas: error bars are
        # TRACE ATTRIBUTES in plotly, so this changes no trace counts and
        # leaves the Long-Term Modulus graft below untouched.
        E_terms, has_eq = _split_terms(tau_i, E_i, covariance)
        sigma_log = sigma_log_coefficients(covariance)
        plus, minus = spectrum_error_bars(
            E_terms, sigma_log[1:] if has_eq else sigma_log)
        spectrum_df["E_err_plus"] = plus
        spectrum_df["E_err_minus"] = minus
        error_kwargs = dict(error_y="E_err_plus", error_y_minus="E_err_minus")
    fig3 = px.scatter(
        spectrum_df, x="Time", y="E",
        log_x=True, log_y=True,
        color="Type", symbol="Type",
        labels={"Time": "Relaxation Time, 𝜏 (s)", "E": "Prony Coefficient, Eᵢ (Pa)"},
        **error_kwargs,
    )
    if solid and E_i[0] > 0:
        fig3.add_trace(go.Scatter(
            x=[tau_i.min(), tau_i.max()],
            y=[E_i[0], E_i[0]],
            mode="lines",
            line=dict(dash="dash"),
            name="Long-Term Modulus",
        ))
    fig3.update_layout(
        autosize=False, margin=dict(l=80, r=60, t=60, b=80),
        legend_title="Type",
    )

    if not fit_settings:
        fig2 = fig2a

    for fig in (fig2, fig3):
        fig.update_xaxes(exponentformat='power')
        fig.update_yaxes(exponentformat='power')
    return fig2, fig3


def _build_coef_records(tau_i: np.ndarray, E_i: np.ndarray,
                        covariance: np.ndarray = None) -> list:
    """
    Build the Prony coefficient table as a list of records.

    Parameters:
        tau_i (numpy.ndarray): Prony relaxation times.
        E_i (numpy.ndarray): Prony coefficients (length tau_i or tau_i + 1).
        covariance (numpy.ndarray): Posterior covariance of the fitted
            log-coefficients, or None to omit the sigma column.

    Returns:
        list: List of dicts with keys 'i', 'tau_i', 'E_i' — one per nonzero
        coefficient, with 'i' the original (pre-filter) index. With a
        covariance, each record also carries 'sigma_log_E_i', the posterior
        1-sigma of ln(E_i) in nepers, capped at _SIGMA_DISPLAY_CAP (a capped
        value reads "unconstrained", and stays finite for the CSV export).
        The column is omitted ENTIRELY when covariance is None, so the
        NNLS/default path keeps its exact schema.
    """
    columns = {"tau_i": tau_i, "E_i": E_i[len(E_i) - len(tau_i):]}
    if covariance is not None:
        # Built BEFORE the nonzero filter below so rows stay aligned.
        E_terms, has_eq = _split_terms(tau_i, E_i, covariance)
        sigma_log = sigma_log_coefficients(covariance)
        columns["sigma_log_E_i"] = np.minimum(
            sigma_log[1:] if has_eq else sigma_log, _SIGMA_DISPLAY_CAP)
    coef_df = pd.DataFrame(columns)
    coef_df = coef_df[coef_df.E_i != 0].reset_index(drop=False)
    coef_df = coef_df.rename(columns={'index': 'i'})
    return coef_df.to_dict("records")


# Rows in the model-only shift table (no measured temperatures to anchor to,
# so the dense evaluation grid is thinned to a CSV-friendly size).
_SHIFT_TABLE_MAX_ROWS = 50


def _build_shift_figure(shiftData, shift_model, Tg, TC, C1, C2, Ea, a_T_ref,
                        data_T_range, chi2_reduced) -> tuple:
    """
    Build the shift-factor figure (a_T vs Temperature, log-y) and its table.

    Draws the uploaded shift factors as markers ("Experiment") when a shift
    file is present, and the WLF or hybrid model as a dashed curve when its
    parameters are complete — either alone is enough for a figure. The curve
    is evaluated on a dense grid spanning the union of the shift file's and
    the viscoelastic data's temperature ranges, and masked to the same
    |log10 a_T| <= MAX_ABS_LOG10_SHIFT window the transform applies, so a
    nearby WLF pole shows as a gap instead of distorting the axis (or, for
    hybrid, raising — see below).

    Parameters:
        shiftData: Optional {'Temperature': ..., 'a_T': ...} mapping from
            upload_init(..., 'shift'); falsy for none.
        shift_model (str): 'WLF', 'hybrid', 'manual', or 'none'/None. Only
            'WLF' and 'hybrid' can draw a model curve.
        Tg, TC, C1, C2, Ea: Shift-model parameters; the curve is skipped
            unless its model's full set is present (Tg/C1/C2 for WLF,
            TC/C1/C2/Ea for hybrid).
        a_T_ref (float): Vertical offset for the model curve — the data's
            shift factor at the model's anchor, co-fitted by
            fit_wlf_coefficients (at Tg) or fit_hybrid_coefficients (at TC).
            None falls back to 1.0, which is right when no shift file was
            fitted (the model is then its own reference) and wrong for a file
            referenced anywhere else.
        data_T_range (tuple): (min, max) temperature of the viscoelastic
            data, or None when no temperature axis exists.
        chi2_reduced (float): Fit-time reduced chi-squared to stamp on the
            figure, or None for no stamp. Passed through from the client
            because only the fit (in /fit-shift/) knows how many parameters
            were free; recomputing here would use a different dof convention.

    Returns:
        tuple: (fig, records) — the figure (empty go.Figure() when neither
        markers nor curve are drawable, matching the other conditional
        figures) and the table rows behind it: at the measured temperatures
        ({'Temperature', 'a_T (measured)', 'a_T (model)'}) when a shift file
        is present, else the masked model grid thinned to
        _SHIFT_TABLE_MAX_ROWS rows of {'Temperature', 'a_T (model)'}. All
        values are Python floats (the route serializes with stdlib json,
        which rejects numpy scalars); 'a_T (model)' is None where the model
        is absent or outside the valid window.
    """
    T_meas = a_meas = None
    if shiftData:
        shift_df = pd.DataFrame(shiftData)
        # The legacy positional shift form has no Temperature column; without
        # one there is nowhere on the T axis to put the markers, so only the
        # model curve (if any) is drawn. The temperature branch synthesizes
        # the pairing before calling, so its positional markers still appear.
        if 'Temperature' in shift_df.columns:
            T_meas = shift_df['Temperature'].to_numpy(dtype=float)
            a_meas = shift_df['a_T'].to_numpy(dtype=float)

    curve_ready = (
        (shift_model == 'WLF'
         and Tg is not None and C1 is not None and C2 is not None)
        or (shift_model == 'hybrid'
            and TC is not None and C1 is not None and C2 is not None
            and Ea is not None)
    )

    def eval_log10(T_arr):
        """log10(a_T) of the model at T_arr; non-finite/out-of-window -> nan."""
        if shift_model == 'WLF':
            with np.errstate(divide='ignore', invalid='ignore'):
                log10_a = wlf_log10_shift(
                    T_arr, Tg, C1, C2,
                    a_T_ref if a_T_ref is not None else 1.0,
                )
        else:
            # hybrid_shift raises ValueError at a hand-entered pole (fitted
            # parameters cannot reach one: C2 is floored at 1 and its WLF
            # branch only sees T > TC). No curve is better than a 500.
            try:
                log10_a = np.log10(hybrid_shift(
                    T_arr, TC, C1, C2, Ea,
                    a_T_ref if a_T_ref is not None else 1.0, True,
                ))
            except ValueError:
                return np.full_like(T_arr, np.nan)
        return np.where(
            np.isfinite(log10_a) & (np.abs(log10_a) <= MAX_ABS_LOG10_SHIFT),
            log10_a, np.nan,
        )

    frames = []
    if T_meas is not None:
        frames.append(pd.DataFrame({
            'Temperature': T_meas, 'a_T': a_meas, 'Type': 'Experiment',
        }))
    model_label = f"{shift_model} fit"
    grid_log10 = None
    if curve_ready:
        spans = [(float(np.min(T_meas)), float(np.max(T_meas)))] \
            if T_meas is not None else []
        if data_T_range is not None:
            spans.append((float(data_T_range[0]), float(data_T_range[1])))
        if spans:
            grid = np.linspace(min(lo for lo, _ in spans),
                               max(hi for _, hi in spans), 200)
            grid_log10 = eval_log10(grid)
            keep = np.isfinite(grid_log10)
            if np.any(keep):
                frames.append(pd.DataFrame({
                    'Temperature': grid[keep],
                    'a_T': 10.0 ** grid_log10[keep],
                    'Type': model_label,
                }))
            else:
                grid_log10 = None

    if not frames:
        return go.Figure(), []

    fig = px.line(
        pd.concat(frames, ignore_index=True),
        x='Temperature', y='a_T',
        log_y=True,
        color='Type', line_dash='Type',
        line_dash_map={'Experiment': 'solid', model_label: 'dash'},
        labels={'Temperature': 'Temperature (C)',
                # Plotly renders HTML in axis titles, so the subscript can be
                # a real capital T (the tab label makes do with Unicode ₜ).
                'a_T': 'Shift Factor, a<sub>T</sub>'},
    )
    fig.update_traces(mode='markers', selector=dict(name='Experiment'))
    fig.update_layout(
        autosize=False, margin=dict(l=80, r=60, t=60, b=80),
        legend_title='Type',
    )
    fig.update_yaxes(exponentformat='power')
    if chi2_reduced is not None:
        _stamp_notice((fig,), f"misfit (χ²/ν) = {chi2_reduced:.3g} | lower is better")

    if T_meas is not None:
        model_at_meas = eval_log10(T_meas) if curve_ready \
            else np.full_like(T_meas, np.nan)
        records = [
            {'Temperature': float(t), 'a_T (measured)': float(m),
             'a_T (model)': None if np.isnan(v) else float(10.0 ** v)}
            for t, m, v in zip(T_meas, a_meas, model_at_meas)
        ]
    else:
        keep = np.isfinite(grid_log10)
        step = max(1, int(np.ceil(np.count_nonzero(keep) / _SHIFT_TABLE_MAX_ROWS)))
        records = [
            {'Temperature': float(t), 'a_T (model)': float(10.0 ** v)}
            for t, v in zip(grid[keep][::step], grid_log10[keep][::step])
        ]
    return fig, records
