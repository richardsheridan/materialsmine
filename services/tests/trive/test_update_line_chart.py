"""
`update_line_chart` — the orchestration layer that turns uploaded data into the
chart figures + coefficient table. Covers the frequency and temperature
domains, the validation / contract-assertion paths, and the std error-column
wiring into smooth_prony_fit.

Some tests load real fixture files from app/trive/files/ (via upload_init),
so this subset touches disk and is slower than the pure-math suites. It does
NOT spin up a Flask app — that's test_routes / test_routes_e2e.

    python -m unittest tests.trive.test_update_line_chart
"""
import unittest
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import numpy as np
from unittest.mock import patch

# Append the directory above 'tests' to sys.path to find the 'app' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.trive.prony import compute_complex, prony_terms_for_span
from app.trive.quality import _FitQuality
from app.trive.fit import (
    smooth_prony_fit,
    _extended_relaxation_space,
    _GRID_EXTENSION_DECADES,
)
from app.trive.shift import wlf_shift, inverse_wlf_shift, inverse_hybrid_shift
from app.trive.tts import MAX_ABS_LOG10_SHIFT
from app.trive.figures import _PLOT_MAX_POINTS
from app.trive.chart import update_line_chart
from app.config import Config
from app.utils.util import upload_init


DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'app', 'trive', 'files',
))


class TestUpdateLineChartFrequency(unittest.TestCase):
    """Characterization tests for the frequency-domain branch of update_line_chart."""

    @classmethod
    def setUpClass(cls):
        Config.FILES_DIRECTORY = DATA_DIR
        cls.uploadData = upload_init(
            'agilus30 (8) master curve 20C.txt', 'frequency',
        )
        cls.N = 10
        # The smoothed path extends the tau grid past the data window
        # (fit._GRID_EXTENSION_DECADES), so figure/table term counts follow
        # the EXTENDED total, computed here from the same helper the fit uses
        # rather than hardcoded.
        freq = np.asarray(cls.uploadData['Frequency'], dtype=float)
        cls.N_total = cls.N + 2 * _extended_relaxation_space(
            1 / freq.max(), 1 / freq.min(), cls.N, 2 * len(freq), True,
            _GRID_EXTENSION_DECADES,
        )[1]
        cls.result = update_line_chart(
            cls.uploadData,
            number_of_prony=cls.N,
            smoothness=0.1,
            fit_settings=True,
            domain='frequency',
        )

    def test_returns_10_tuple(self):
        self.assertEqual(len(self.result), 10)
        # Element 9 is the rank_info dict (see update_line_chart's Returns).
        self.assertEqual(
            set(self.result[9]), {'max_prony', 'noise_prony'}
        )

    def test_coef_df_schema(self):
        # Smoothed fit with a covariance: the table carries the sigma column
        # (a flat float per row — the frontend renders/exports rows as-is).
        coef_df = self.result[6]
        self.assertIsInstance(coef_df, list)
        self.assertGreater(len(coef_df), 0)
        for row in coef_df:
            self.assertSetEqual(
                set(row.keys()), {'i', 'tau_i', 'E_i', 'sigma_log_E_i'})
            self.assertNotEqual(row['E_i'], 0.0)
            self.assertIsInstance(row['sigma_log_E_i'], float)
            self.assertTrue(np.isfinite(row['sigma_log_E_i']))
            self.assertGreater(row['sigma_log_E_i'], 0.0)
        # 'i' values are the original DataFrame indices, all nonnegative
        self.assertTrue(all(row['i'] >= 0 for row in coef_df))

    def test_fig1_trace_names(self):
        fig1 = self.result[0]
        names = [t.name for t in fig1.data]
        # Two facets (E Storage, E Loss) × two colors (Experiment + N-Term Prony)
        self.assertEqual(names.count('Experiment'), 2)
        self.assertEqual(sum(1 for n in names if 'Term Prony' in n), 2)

    def test_fig1_experiment_y_matches_input(self):
        # px.line(facet_col='Modulus') splits by Modulus, so the two Experiment
        # traces carry the E Storage and E Loss columns from the input.
        fig1 = self.result[0]
        experiment_y = sorted(
            (tuple(t.y) for t in fig1.data if t.name == 'Experiment'),
            key=lambda ys: ys[0],
        )
        expected = sorted(
            (tuple(self.uploadData['E Loss']), tuple(self.uploadData['E Storage'])),
            key=lambda ys: ys[0],
        )
        for got, want in zip(experiment_y, expected):
            np.testing.assert_array_equal(got, want)

    def test_fig2_trace_counts_with_fit_settings_true(self):
        # fit_settings=True → fig2 is an overlay fig (line + basis scatter),
        # preceded by the ±1σ band pair on this smoothed fixture.
        fig2 = self.result[2]
        self.assertEqual(len(fig2.data), 4)
        names2 = {t.name for t in fig2.data}
        self.assertTrue(any('Basis' in n for n in names2))
        self.assertEqual(
            [t.name for t in fig2.data[:2]], ['±1σ', '±1σ'])

    def test_fig3_is_discrete_spectrum_dot_plot(self):
        # fig3 is the discrete relaxation spectrum: the Prony coefficients as
        # a marker trace at (tau_i, E_i), plus a horizontal dashed line at the
        # long-term (equilibrium) modulus. No Alfrey-style continuous spectrum.
        fig3 = self.result[3]
        self.assertEqual(len(fig3.data), 2)
        names3 = [t.name for t in fig3.data]
        self.assertFalse(any('Basis' in n for n in names3))
        dots = next(t for t in fig3.data if 'Term Prony' in t.name)
        self.assertEqual(dots.mode, 'markers')
        self.assertEqual(len(dots.x), self.N_total)
        # Every figure labels the decaying terms only — the equilibrium
        # coefficient is a separate parameter, drawn here as its own trace — so
        # this label matches the one on the E(t) figure exactly.
        fig2_prony = next(t.name for t in self.result[2].data
                          if 'Term Prony' in (t.name or ''))
        decaying = int(fig2_prony.split('-')[0])
        self.assertEqual(dots.name, f'{decaying}-Term Prony')
        hline = next(t for t in fig3.data if t.name == 'Long-Term Modulus')
        self.assertEqual(hline.mode, 'lines')
        self.assertEqual(len(hline.y), 2)
        self.assertEqual(hline.y[0], hline.y[1])
        self.assertGreater(hline.y[0], 0)

    def test_fig11_tan_delta_ticks_sit_on_the_outside_edge(self):
        # tan delta cannot share the modulus panel's scale, so its facet keeps
        # its own tick labels; on the default left side they overlap the plot
        # to their left. fig1's facets DO share a scale, so its second axis
        # draws no labels and needs no such treatment.
        fig1, fig11 = self.result[0], self.result[1]
        self.assertTrue(fig11.layout.yaxis2.showticklabels)
        self.assertEqual(fig11.layout.yaxis2.side, 'right')
        self.assertFalse(fig1.layout.yaxis2.showticklabels)
        # Those labels land in the legend's lane — automargin reserves room for
        # the legend but not for them — so the legend clears its 1.02 default.
        self.assertGreater(fig11.layout.legend.x, 1.02)
        self.assertIsNone(fig1.layout.legend.x)

    def test_fig4_fig41_empty_without_a_transform_request(self):
        # The class fixture asks for no shift model, so the temperature axis —
        # a transform of the upload rather than the upload itself — is not
        # drawn. Previously it was synthesized from universal-WLF constants.
        fig4, fig41 = self.result[4], self.result[5]
        for fig in (fig4, fig41):
            self.assertEqual(len(fig.data), 0)

    def test_fig4_fig41_have_only_experiment_traces(self):
        # In frequency domain, fig4/fig41 visualize the inverse-WLF temperature
        # conversion of the input — no Prony fit is overlaid there.
        result = update_line_chart(
            self.uploadData, number_of_prony=self.N, smoothness=0.1,
            fit_settings=True, domain='frequency',
            shift_model='WLF', Tg=20.0, C1=17.44, C2=51.6,
        )
        fig4, fig41 = result[4], result[5]
        for fig in (fig4, fig41):
            names = {t.name for t in fig.data}
            self.assertEqual(names, {'Experiment'})

    def test_fit_settings_false_drops_fig2_basis_overlay_only(self):
        result = update_line_chart(
            self.uploadData, number_of_prony=self.N, smoothness=0.1,
            fit_settings=False, domain='frequency',
        )
        fig2, fig3 = result[2], result[3]
        # Band pair + the fit line; the basis overlay alone is dropped.
        self.assertEqual(len(fig2.data), 3)
        self.assertNotIn('Basis', {t.name for t in fig2.data})
        # fig3 is the discrete-spectrum dot plot regardless of fit_settings.
        self.assertEqual(
            [t.name for t in fig3.data],
            [t.name for t in self.result[3].data],
        )


class TestUpdateLineChartFrequencyShift(unittest.TestCase):
    """
    Frequency-domain shift-model paths (manual / WLF / hybrid) that drive the
    temperature-axis visualization (fig4/fig41). Unusable inputs RAISE out of
    tts_frequency_to_temperature_V2 (no silent universal-WLF fallback), so a
    requested-but-broken transform blocks the response like any other input
    error; only an unrequested transform leaves the fit standing alone.
    """

    @classmethod
    def setUpClass(cls):
        Config.FILES_DIRECTORY = DATA_DIR
        cls.uploadData = upload_init(
            'agilus30 (8) master curve 20C.txt', 'frequency',
        )
        # Monotonic synthetic shift table; a_T decreasing through 1.0 at T = 30.
        T = np.linspace(-20.0, 80.0, 21)
        cls.shiftData = {'Temperature': T, 'a_T': 10.0 ** np.linspace(6.0, -6.0, len(T))}

    def _run(self, **kw):
        return update_line_chart(
            self.uploadData, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='frequency', **kw,
        )

    @staticmethod
    def _fig4_temps(result):
        # fig4 is px.line(x="Temperature", facet_col='Modulus'); both facets carry
        # the same temperature axis, so dedupe to the underlying sorted set.
        fig4 = result[4]
        return np.unique(np.concatenate([np.asarray(t.x, float) for t in fig4.data]))

    def test_frequency_manual_uses_shiftData(self):
        manual = self._run(shift_model='manual', shiftData=self.shiftData)
        wlf = self._run(shift_model='WLF', Tg=30.0, C1=17.44, C2=51.6)
        self.assertEqual(len(manual), 10)
        # The manual mapping reached fig4: its temperature axis differs from a
        # WLF evaluation (a different shape alone already proves it, since
        # np.interp clamps).
        mt, dt = self._fig4_temps(manual), self._fig4_temps(wlf)
        self.assertFalse(mt.shape == dt.shape and np.allclose(mt, dt))

    def test_frequency_WLF_populates_temp_figs(self):
        Tg, C1, C2 = 20.0, 17.44, 51.6
        result = self._run(shift_model='WLF', Tg=Tg, C1=C1, C2=C2)
        omega = self.uploadData['Frequency']
        expected = np.unique(inverse_wlf_shift(omega / 1.0, Tg, C1, C2))
        np.testing.assert_allclose(self._fig4_temps(result), expected, rtol=1e-6)

    def test_frequency_hybrid_populates_temp_figs(self):
        TC, C1, C2, Ea = 20.0, 17.44, 51.6, 200.0
        result = self._run(shift_model='hybrid', TC=TC, C1=C1, C2=C2, Ea=Ea)
        omega = self.uploadData['Frequency']
        expected = np.unique(inverse_hybrid_shift(omega / 1.0, TC, C1, C2, Ea))
        np.testing.assert_allclose(self._fig4_temps(result), expected, rtol=1e-6)

    def test_frequency_hybrid_missing_Ea_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(shift_model='hybrid', TC=20.0, C1=17.44, C2=51.6)
        self.assertIn('Ea', str(ctx.exception))

    def test_frequency_manual_without_shiftData_raises(self):
        # Same contract as the temperature branch's forward transform: manual
        # without a file is a user error, not a silent degradation.
        with self.assertRaises(ValueError) as ctx:
            self._run(shift_model='manual', shiftData=None)
        self.assertIn('no shift-factor file', str(ctx.exception))

    def test_frequency_WLF_missing_Tg_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(shift_model='WLF', C1=17.44, C2=51.6)
        self.assertIn('Tg', str(ctx.exception))

    def test_frequency_temp_view_carries_provenance_caption(self):
        # The "labeled" half of the contract: the synthesized temperature axis
        # names the inverse that produced it.
        wlf = self._run(shift_model='WLF', Tg=30.0, C1=17.44, C2=51.6)
        hybrid = self._run(shift_model='hybrid', TC=20.0, C1=17.44, C2=51.6,
                           Ea=200.0)
        manual = self._run(shift_model='manual', shiftData=self.shiftData)
        for result, needle in ((wlf, 'inverse WLF at Tg = 30'),
                               (hybrid, 'inverse hybrid at Tc = 20'),
                               (manual, 'uploaded shift factors')):
            for fig in (result[4], result[5]):
                texts = [a.text for a in fig.layout.annotations if a.text]
                self.assertTrue(any(needle in t for t in texts),
                                f'missing "{needle}" in {texts}')

    def test_frequency_none_suppresses_temp_figs_but_not_the_fit(self):
        # 'none' means the caller did not ask for a transform, so the
        # temperature axis is not drawn — but the Prony fit runs on the
        # frequency data directly and is unaffected.
        result = self._run(shift_model='none')
        self.assertEqual(len(result[4].data), 0)
        self.assertEqual(len(result[5].data), 0)
        self.assertGreater(len(result[0].data), 0)  # fig1 still built
        self.assertGreater(len(result[6]), 0)       # coef_df non-empty

    def test_frequency_unspecified_shift_model_matches_none(self):
        # The Python default (None) and the wire value ('none') mean the same
        # thing: nothing was requested.
        omitted, explicit = self._run(), self._run(shift_model='none')
        for fig_idx in (4, 5):
            self.assertEqual(len(omitted[fig_idx].data), 0)
            self.assertEqual(len(explicit[fig_idx].data), 0)

    def test_frequency_none_with_shiftData_still_builds_temp_figs(self):
        # A shift table is itself a request for a transform, so it wins over
        # a 'none' model rather than being silently discarded.
        result = self._run(shift_model='none', shiftData=self.shiftData)
        self.assertGreater(len(result[4].data), 0)
        self.assertGreater(len(result[5].data), 0)

    def test_prony_fit_unaffected_by_shift_params(self):
        def exp_y(result):
            fig1 = result[0]
            return sorted(
                (tuple(t.y) for t in fig1.data if t.name == 'Experiment'),
                key=lambda ys: ys[0],
            )
        with_shift = self._run(shift_model='manual', shiftData=self.shiftData)
        without = self._run()
        for got, want in zip(exp_y(with_shift), exp_y(without)):
            np.testing.assert_array_equal(got, want)


class TestUpdateLineChartShiftFigure(unittest.TestCase):
    """
    The shift-factor figure and table (elements 7 and 8 of the return):
    measured markers, model curve, pole masking, the chi2 stamp, and the
    empty-figure conventions.
    """

    T_REF = 25.0
    C1 = 17.44
    C2 = 51.6

    @classmethod
    def setUpClass(cls):
        Config.FILES_DIRECTORY = DATA_DIR
        T = np.linspace(0.0, 80.0, 30)
        cls.temp_data = {
            'Temperature': T,
            'E Storage': np.linspace(1000.0, 10.0, len(T)),
            'E Loss': np.full(len(T), 50.0),
        }
        Ts = np.linspace(0.0, 80.0, 9)
        cls.shiftData = {
            'Temperature': Ts,
            'a_T': 10.0 ** np.linspace(3.0, -3.0, len(Ts)),
        }

    def _run(self, **kw):
        return update_line_chart(
            self.temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature', **kw,
        )

    @staticmethod
    def _trace(fig, name):
        matches = [t for t in fig.data if t.name == name]
        assert len(matches) == 1, [t.name for t in fig.data]
        return matches[0]

    def test_no_transform_yields_empty_figure_and_table(self):
        # Temperature early-exit (no shift params) — nothing to draw.
        *_, shift_fig, shift_records, _rank = self._run()
        self.assertEqual(len(shift_fig.data), 0)
        self.assertEqual(shift_records, [])

    def test_frequency_none_yields_empty_figure_and_table(self):
        freq_data = upload_init(
            'agilus30 (8) master curve 20C.txt', 'frequency')
        *_, shift_fig, shift_records, _rank = update_line_chart(
            freq_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='frequency', shift_model='none',
        )
        self.assertEqual(len(shift_fig.data), 0)
        self.assertEqual(shift_records, [])

    def test_manual_draws_markers_only(self):
        # 'manual' has no model curve of its own — Experiment markers only.
        *_, shift_fig, shift_records, _rank = self._run(
            shift_model='manual', shiftData=self.shiftData)
        self.assertEqual([t.name for t in shift_fig.data], ['Experiment'])
        self.assertEqual(self._trace(shift_fig, 'Experiment').mode, 'markers')
        # Table rows carry the measured values; no model column values.
        self.assertEqual(len(shift_records), len(self.shiftData['a_T']))
        self.assertTrue(all(r['a_T (model)'] is None for r in shift_records))
        self.assertEqual(shift_records[0]['a_T (measured)'],
                         float(self.shiftData['a_T'][0]))

    def test_wlf_with_shift_file_draws_markers_and_dashed_curve(self):
        *_, shift_fig, shift_records, _rank = self._run(
            shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=self.C2,
            shiftData=self.shiftData)
        self.assertEqual({t.name for t in shift_fig.data},
                         {'Experiment', 'WLF fit'})
        curve = self._trace(shift_fig, 'WLF fit')
        self.assertEqual(curve.line.dash, 'dash')
        self.assertEqual(self._trace(shift_fig, 'Experiment').mode, 'markers')
        # Model column now populated at the measured temperatures, and it
        # matches the WLF equation there — except where the model leaves the
        # |log10 a_T| <= MAX_ABS_LOG10_SHIFT window (the coldest points here),
        # which are masked to None exactly as the transform drops those rows.
        expected = wlf_shift(np.asarray(self.shiftData['Temperature']),
                             self.T_REF, self.C1, self.C2)
        in_window = np.abs(np.log10(expected)) <= MAX_ABS_LOG10_SHIFT
        self.assertTrue(in_window.any() and not in_window.all())
        got = [r['a_T (model)'] for r in shift_records]
        for g, e, ok in zip(got, expected, in_window):
            if ok:
                self.assertAlmostEqual(g / e, 1.0, places=6)
            else:
                self.assertIsNone(g)

    def test_model_only_wlf_draws_curve_over_data_range(self):
        *_, shift_fig, shift_records, _rank = self._run(
            shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=self.C2)
        self.assertEqual([t.name for t in shift_fig.data], ['WLF fit'])
        curve = self._trace(shift_fig, 'WLF fit')
        # The grid spans the data's temperature range, minus the cold points
        # the |log10 a_T| window masks off; the warm end is inside the window.
        self.assertGreaterEqual(min(curve.x), 0.0)
        self.assertLess(min(curve.x), 5.0)
        self.assertAlmostEqual(max(curve.x), 80.0)
        y = np.asarray(curve.y, dtype=float)
        self.assertTrue(np.all(np.abs(np.log10(y)) <= MAX_ABS_LOG10_SHIFT))
        # Model-only table is the thinned grid.
        self.assertLessEqual(len(shift_records), 50)
        self.assertTrue(all(set(r) == {'Temperature', 'a_T (model)'}
                            for r in shift_records))

    def test_wlf_curve_honors_a_T_ref_offset(self):
        # The WLF curve carries the same co-fitted offset the hybrid one does;
        # without it a fit against a table referenced away from Tg draws a
        # curve parallel to — and decades off — its own Experiment markers.
        kwargs = dict(shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=self.C2)
        *_, fig_unit, _, _rank = self._run(**kwargs)
        *_, fig_shifted, _, _rank = self._run(a_T_ref=100.0, **kwargs)
        unit = self._trace(fig_unit, 'WLF fit')
        shifted = self._trace(fig_shifted, 'WLF fit')
        # Compare at shared temperatures, not by position: the offset moves
        # which grid points clear the |log10 a_T| window, so the two traces
        # are drawn over different (overlapping) spans of the same grid.
        common, i_unit, i_shift = np.intersect1d(
            np.array(unit.x), np.array(shifted.x), return_indices=True)
        self.assertGreater(len(common), 0)
        np.testing.assert_allclose(
            np.array(shifted.y)[i_shift] / np.array(unit.y)[i_unit],
            100.0, rtol=1e-9)

    def test_hybrid_curve_honors_a_T_ref_offset(self):
        kwargs = dict(shift_model='hybrid', TC=40.0, C1=self.C1, C2=self.C2,
                      Ea=150.0)
        *_, fig_unit, _, _rank = self._run(**kwargs)
        *_, fig_shifted, _, _rank = self._run(a_T_ref=100.0, **kwargs)
        y_unit = np.array(self._trace(fig_unit, 'hybrid fit').y)
        y_shifted = np.array(self._trace(fig_shifted, 'hybrid fit').y)
        # Same grid, curve multiplied by the offset (where both are drawn).
        n = min(len(y_unit), len(y_shifted))
        np.testing.assert_allclose(y_shifted[:n] / y_unit[:n], 100.0,
                                   rtol=1e-9)

    def test_shift_reference_draws_markers_without_driving_transform(self):
        # A display-only reference file: Experiment markers appear alongside
        # the model curve, but the transform must still come from the model —
        # unlike shiftData, whose table would override it.
        kwargs = dict(shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=self.C2)
        with_ref = self._run(shift_reference=self.shiftData, **kwargs)
        without = self._run(**kwargs)
        fig_ref = with_ref[7]
        self.assertEqual({t.name for t in fig_ref.data},
                         {'Experiment', 'WLF fit'})
        # Identical master curves prove the reference never reached the
        # transform (the interpolated table would shift the frequencies).
        np.testing.assert_array_equal(
            with_ref[0].data[0].x, without[0].data[0].x)

    def test_shiftdata_wins_over_shift_reference_for_markers(self):
        # When both are present the applied table is the honest marker source.
        ref = {'Temperature': np.array([10.0, 20.0]),
               'a_T': np.array([123.0, 1.0])}
        *_, shift_fig, shift_records, _rank = self._run(
            shift_model='manual', shiftData=self.shiftData,
            shift_reference=ref)
        self.assertEqual(len(shift_records), len(self.shiftData['a_T']))

    def test_wlf_pole_in_range_is_masked_not_raised(self):
        # Fixed C2 puts the pole at Tg - C2 = 15, inside the 0-80 data range.
        # The transform itself would raise; the figure must instead draw the
        # valid window and skip the rest — so use a shift file to carry the
        # transform and hand the curve bad parameters.
        *_, shift_fig, _, _rank = self._run(
            shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=10.0,
            shiftData=self.shiftData)
        curve = self._trace(shift_fig, 'WLF fit')
        y = np.asarray(curve.y, dtype=float)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.abs(np.log10(y)) <= MAX_ABS_LOG10_SHIFT))

    def test_chi2_stamp_present_iff_passed(self):
        kwargs = dict(shift_model='WLF', Tg=self.T_REF, C1=self.C1,
                      C2=self.C2, shiftData=self.shiftData)

        def notices(fig):
            return [a.text for a in (fig.layout.annotations or ())
                    if getattr(a, 'name', '') == 'figure-notice']

        *_, without, _, _rank = self._run(**kwargs)
        self.assertEqual(notices(without), [])
        *_, with_stamp, _, _rank = self._run(shift_chi2_reduced=0.123, **kwargs)
        self.assertEqual(notices(with_stamp),
                         ['misfit (χ²/ν) = 0.123 | lower is better'])

    def test_figure_and_table_survive_json_round_trip(self):
        import json as _json
        *_, shift_fig, shift_records, _rank = self._run(
            shift_model='WLF', Tg=self.T_REF, C1=self.C1, C2=self.C2,
            shiftData=self.shiftData, shift_chi2_reduced=0.5)
        blob = _json.dumps({'shift-chart': _json.loads(shift_fig.to_json()),
                            'shift-table': shift_records})
        self.assertIn('"shift-table"', blob)


class TestUpdateLineChartTemperature(unittest.TestCase):
    """Characterization tests for the temperature-domain branches."""

    T_REF = 25.0
    C1 = 17.44
    C2 = 51.6
    EA = 200.0

    @classmethod
    def setUpClass(cls):
        Config.FILES_DIRECTORY = DATA_DIR
        # Real temperature ramp for the early-exit path.
        cls.real_temp_data = upload_init(
            'agilus30 (8) Temperature Ramp clean.txt', 'temperature',
        )
        # Synthetic temperature ramp chosen so WLF and hybrid shifts both stay
        # well-defined (T spans both sides of T_ref, well clear of the
        # WLF C2 singularity at T_ref - C2).
        T = np.linspace(0.0, 80.0, 30)
        cls.synthetic_temp_data = {
            'Temperature': T,
            'E Storage': np.linspace(1000.0, 10.0, len(T)),
            'E Loss': np.full(len(T), 50.0),
        }

    def test_early_exit_with_no_shift_params(self):
        (fig1, fig11, fig2, fig3, fig4, fig41, coef_df,
         shift_fig, shift_records, rank_info) = update_line_chart(
            self.real_temp_data, number_of_prony=10, smoothness=0.1,
            fit_settings=True, domain='temperature',
        )
        for empty in (fig1, fig11, fig2, fig3):
            self.assertEqual(len(empty.data), 0)
        self.assertEqual(coef_df, [])
        # No master curve was built, so there is nothing to rank-probe:
        # None, not a dict of Nones.
        self.assertIsNone(rank_info)
        self.assertGreater(len(fig4.data), 0)
        self.assertGreater(len(fig41.data), 0)
        # Same unshared-tan-delta axis as the frequency figure: outside edge,
        # with the legend moved off it.
        self.assertTrue(fig41.layout.yaxis2.showticklabels)
        self.assertEqual(fig41.layout.yaxis2.side, 'right')
        self.assertGreater(fig41.layout.legend.x, 1.02)
        self.assertFalse(fig4.layout.yaxis2.showticklabels)
        self.assertIsNone(fig4.layout.legend.x)

    def test_WLF_shift_populates_all_figures(self):
        result = update_line_chart(
            self.synthetic_temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            Tg=self.T_REF, C1=self.C1, C2=self.C2, shift_model='WLF',
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertIsInstance(coef_df, list)
        self.assertGreater(len(coef_df), 0)

    def test_hybrid_shift_populates_all_figures(self):
        result = update_line_chart(
            self.synthetic_temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            Tg=self.T_REF, TC=self.T_REF, C1=self.C1, C2=self.C2,
            Ea=self.EA, shift_model='hybrid',
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertGreater(len(coef_df), 0)

    def test_hybrid_tolerates_duplicate_temperature_rows(self):
        # upload_init keeps duplicate-temperature rows (the bundled VeroCyan
        # ramp contains one), and hybrid_shift used to reject the tie with an
        # AssertionError that escaped the route as a 500 instead of the usual
        # 400-with-message.
        data = {
            k: np.append(np.asarray(v), np.asarray(v)[-1])
            for k, v in self.synthetic_temp_data.items()
        }
        result = update_line_chart(
            data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            TC=self.T_REF, C1=self.C1, C2=self.C2,
            Ea=self.EA, shift_model='hybrid',
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertGreater(len(coef_df), 0)

    def test_hybrid_works_without_Tg(self):
        # hybrid_shift uses TC (not Tg) as the WLF/Arrhenius crossover, so
        # update_line_chart should drive the master-curve path even when Tg is
        # not supplied.
        result = update_line_chart(
            self.synthetic_temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            TC=self.T_REF, C1=self.C1, C2=self.C2, Ea=self.EA,
            shift_model='hybrid',
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertGreater(len(coef_df), 0)

    def test_Tg_zero_does_not_falsely_trigger_early_exit(self):
        # Regression: the old `Tg and C1 and C2` truthy check treated Tg = 0 °C
        # as missing and dropped users into the empty-figures branch.
        result = update_line_chart(
            self.synthetic_temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            Tg=0.0, C1=self.C1, C2=self.C2, shift_model='WLF',
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertGreater(len(coef_df), 0)

    def test_shiftData_override_triggers_shift_path(self):
        # Even without Tg/C1/C2/shift_model, supplying shiftData should drive
        # the shift branch (b is true via the shiftData term).
        n = len(self.synthetic_temp_data['Temperature'])
        T = self.synthetic_temp_data['Temperature']
        a_T = wlf_shift(T, self.T_REF, self.C1, self.C2)
        shiftData = {'Temperature': T, 'a_T': a_T}
        result = update_line_chart(
            self.synthetic_temp_data, number_of_prony=8, smoothness=0.1,
            fit_settings=True, domain='temperature',
            shiftData=shiftData,
        )
        fig1, fig11, fig2, fig3, fig4, fig41, coef_df, _, _, _ = result
        for fig in (fig1, fig11, fig2, fig3, fig4, fig41):
            self.assertGreater(len(fig.data), 0)
        self.assertGreater(len(coef_df), 0)


class TestUpdateLineChartUncertaintyBands(unittest.TestCase):
    """The ±1σ display: ribbons, error bars, the sigma column — and their
    clean absence on the paths that have no covariance."""

    @staticmethod
    def _upload():
        tau = np.logspace(-4.0, 4.0, 9)
        E_input = np.concatenate(
            ([1e6], np.exp(-(np.log10(tau)) ** 2 / 4.0) * 1e9))
        df = compute_complex(tau, E_input, num_pts=200)
        return {
            'Frequency': df['Frequency'].to_numpy(),
            'E Storage': df['E Storage'].to_numpy(),
            'E Loss': df['E Loss'].to_numpy(),
        }

    def _run(self, smoothness):
        return update_line_chart(
            self._upload(), number_of_prony=20, smoothness=smoothness,
            fit_settings=True, domain='frequency',
        )

    @staticmethod
    def _bands(fig):
        return [t for t in fig.data if t.name == '±1σ']

    def test_smoothed_run_draws_ribbons_under_the_curves(self):
        fig1, fig11, fig2 = self._run(0.1)[:3]
        for fig, n_pairs in ((fig1, 2), (fig11, 2), (fig2, 1)):
            bands = self._bands(fig)
            self.assertEqual(len(bands), 2 * n_pairs)
            # Every ribbon trace precedes every px trace, so the bands draw
            # under both the experiment and fit lines.
            self.assertEqual([t.name for t in fig.data[:2 * n_pairs]],
                             ['±1σ'] * 2 * n_pairs)
            # Exactly one legend entry toggles them all (shared legendgroup).
            self.assertEqual(sum(bool(t.showlegend) for t in bands), 1)
            self.assertEqual({t.legendgroup for t in bands}, {'±1σ'})
            # Pairs stay adjacent: fill='tonexty' binds to the PREVIOUS trace
            # in data order, so it sits on the 2nd trace of each pair only.
            for k, t in enumerate(bands):
                self.assertEqual(t.fill, 'tonexty' if k % 2 else None)

    def test_facet_axis_assignment(self):
        # Col 1 (E Storage) on ('x','y'), col 2 (E Loss / tan delta) on
        # ('x2','y2') — explicit, or the fill would leak across facets.
        for fig in self._run(0.1)[:2]:
            self.assertEqual(
                [(t.xaxis, t.yaxis) for t in self._bands(fig)],
                [('x', 'y'), ('x', 'y'), ('x2', 'y2'), ('x2', 'y2')])

    def test_log_panel_lower_edges_stay_positive(self):
        # The harmonic lower edge y²/(y+σ): a log axis must never see 0.
        fig1, _, fig2 = self._run(0.1)[:3]
        for fig in (fig1, fig2):
            for lower in self._bands(fig)[::2]:
                y = np.asarray(lower.y, dtype=float)
                self.assertTrue(np.isfinite(y).all())
                self.assertTrue((y > 0).all())

    def test_fig3_error_bars_present_finite_and_asymmetric(self):
        fig3 = self._run(0.1)[3]
        dots = next(t for t in fig3.data if 'Term Prony' in t.name)
        plus = np.asarray(dots.error_y.array, dtype=float)
        minus = np.asarray(dots.error_y.arrayminus, dtype=float)
        E = np.asarray(dots.y, dtype=float)
        self.assertEqual(len(plus), len(E))
        self.assertTrue(np.isfinite(plus).all())
        self.assertTrue(np.isfinite(minus).all())
        self.assertTrue((plus >= minus).all())  # log-normal asymmetry
        self.assertTrue((minus < E).all())      # lower edge stays positive
        # The Long-Term Modulus reference line carries no bar (deliberate).
        hline = next((t for t in fig3.data if t.name == 'Long-Term Modulus'),
                     None)
        if hline is not None:
            self.assertIsNone(hline.error_y.array)

    def test_unsmoothed_run_has_no_uncertainty_display(self):
        fig1, fig11, fig2, fig3, _, _, coef_df, _, _, _ = self._run(0.0)
        for fig in (fig1, fig11, fig2, fig3):
            self.assertEqual(self._bands(fig), [])
        dots = next(t for t in fig3.data if 'Term Prony' in t.name)
        self.assertIsNone(dots.error_y.array)
        for row in coef_df:
            self.assertNotIn('sigma_log_E_i', row)

    def test_cholesky_failure_degrades_gracefully(self):
        # No covariance -> exactly today's band-free output, no exception:
        # the graceful-degradation guarantee behind covariance=None.
        with patch('app.trive.quality._cholesky_or_none', return_value=None):
            fig1, fig11, fig2, fig3, _, _, coef_df, _, _, _ = self._run(0.1)
        for fig in (fig1, fig11, fig2, fig3):
            self.assertEqual(self._bands(fig), [])
        for row in coef_df:
            self.assertNotIn('sigma_log_E_i', row)


class TestUpdateLineChartTermLabels(unittest.TestCase):
    """
    Every "N-Term Prony" label counts decaying terms only. The equilibrium
    coefficient is a separate parameter — no tau_i, exempt from the smoothness
    penalty, split into its own trace on fig3, dropped from the coefficient
    table — so it must not appear in any term count.
    """

    @staticmethod
    def _upload():
        # Nine-mode source, densely sampled: NNLS has real structure to select
        # from, so the unsmoothed path lands well short of the grid size.
        tau = np.logspace(-4.0, 4.0, 9)
        E_input = np.concatenate(
            ([1e6], np.exp(-(np.log10(tau)) ** 2 / 4.0) * 1e9))
        df = compute_complex(tau, E_input, num_pts=200)
        return {
            'Frequency': df['Frequency'].to_numpy(),
            'E Storage': df['E Storage'].to_numpy(),
            'E Loss': df['E Loss'].to_numpy(),
        }

    @staticmethod
    def _label_counts(figs):
        return {int(t.name.split('-')[0])
                for fig in figs for t in fig.data if 'Term Prony' in t.name}

    def _run(self, N, smoothness):
        return update_line_chart(
            self._upload(), number_of_prony=N, smoothness=smoothness,
            fit_settings=True, domain='frequency',
        )

    def _extended_total(self, N):
        # On the smoothed path labels count the EXTENDED grid, so the slider
        # value no longer numerically matches the label; compute the expected
        # total from the same helper the fit uses.
        freq = self._upload()['Frequency']
        return N + 2 * _extended_relaxation_space(
            1 / freq.max(), 1 / freq.min(), N, 2 * len(freq), True,
            _GRID_EXTENSION_DECADES,
        )[1]

    def test_smoothed_labels_match_the_coefficient_table(self):
        # Regression: on this path every coefficient is exp(...) and so never
        # exactly zero, which made the old count_nonzero over the whole vector
        # report the grid size plus the equilibrium term — one more term than
        # the grid whose rows the table listed.
        fig1, fig11, fig2, fig3, _, _, coef_df, _, _, _ = self._run(23, 0.04)
        expected = self._extended_total(23)
        self.assertGreater(expected, 23)  # the extension is actually on
        self.assertEqual(len(coef_df), expected)
        self.assertEqual(
            self._label_counts((fig1, fig11, fig2, fig3)), {expected})

    def test_unsmoothed_labels_match_the_coefficient_table(self):
        # NNLS zeroes coefficients outright, so here the count is genuinely
        # below the grid size — and still must not pick up the equilibrium term.
        fig1, fig11, fig2, fig3, _, _, coef_df, _, _, _ = self._run(23, 0.0)
        self.assertLess(len(coef_df), 23)
        self.assertEqual(self._label_counts((fig1, fig11, fig2, fig3)),
                         {len(coef_df)})

    def test_basis_overlay_label_matches_too(self):
        # fig2's basis scatter is drawn over tau_i, which has no equilibrium
        # entry either.
        fig2 = self._run(23, 0.04)[2]
        basis = [t.name for t in fig2.data if 'Term Basis' in t.name]
        self.assertEqual(basis, [f'{self._extended_total(23)}-Term Basis'])


class TestUpdateLineChartTemperaturePronyTerms(unittest.TestCase):
    """
    A temperature upload has no frequency axis for the client to measure, so it
    sends the store default of PRONY_TERMS_MAX terms however short the ramp.
    update_line_chart sizes the series against the master curve the ω-T
    transform produces and treats the request as a ceiling.
    """

    T_REF = 25.0
    C1 = 17.44
    C2 = 51.6

    # Short ramp, in the spirit of the bundled 1 Hz temperature files (13 and 19
    # rows): far too few points to carry 100 Prony terms.
    SHORT_RAMP = {
        'Temperature': np.linspace(0.0, 80.0, 13),
        'E Storage': np.linspace(1000.0, 10.0, 13),
        'E Loss': np.full(13, 50.0),
    }

    def _run(self, number_of_prony, data=None, **kwargs):
        return update_line_chart(
            data if data is not None else self.SHORT_RAMP,
            number_of_prony=number_of_prony, smoothness=0.1,
            fit_settings=False, domain='temperature',
            Tg=self.T_REF, C1=self.C1, C2=self.C2, shift_model='WLF',
            **kwargs,
        )

    @staticmethod
    def _spy():
        return patch(
            'app.trive.chart.smooth_prony_fit',
            return_value=(np.array([1.0]), np.array([1.0]),
                          _FitQuality(1.0, 2.0, 3.0)),
        )

    @staticmethod
    def _quality_readouts(fig):
        return [a.text for a in fig.layout.annotations
                if a.text and 'lower is better' in a.text]

    def test_request_is_capped_by_the_transformed_span(self):
        with self._spy() as spy:
            self._run(100)
        N = spy.call_args.kwargs['N']
        omega = spy.call_args.kwargs['omega']
        self.assertLess(N, 100)
        self.assertEqual(N, prony_terms_for_span(omega))

    def test_lower_request_is_honored_unchanged(self):
        # The ceiling must not become a replacement: turning the term count
        # down is still the user's call in this domain.
        with self._spy() as spy:
            self._run(4)
        self.assertEqual(spy.call_args.kwargs['N'], 4)

    def test_short_ramp_at_the_default_still_reports_a_misfit(self):
        # Regression: at N = 100 the series carried more parameters than a
        # 13-row ramp has residuals, so chi-squared had no degrees of freedom,
        # _prony_fit_quality returned None for it, and _annotate_fit_quality
        # dropped it — the readout showed curvature and surprisal only.
        fig1, fig11 = self._run(100)[:2]
        for fig in (fig1, fig11):
            readouts = self._quality_readouts(fig)
            self.assertEqual(len(readouts), 1)
            self.assertIn('χ²/ν', readouts[0])

    def test_frequency_domain_request_is_untouched(self):
        # The client already sizes the series from the file itself there, so an
        # explicit term count is passed through as given.
        freq_data = {
            'Frequency': np.logspace(-2, 2, 13),
            'E Storage': np.linspace(1000.0, 10.0, 13),
            'E Loss': np.full(13, 50.0),
        }
        with self._spy() as spy:
            update_line_chart(
                freq_data, number_of_prony=100, smoothness=0.1,
                fit_settings=False, domain='frequency',
            )
        self.assertEqual(spy.call_args.kwargs['N'], 100)


class TestUpdateLineChartValidation(unittest.TestCase):
    """
    Validation paths in update_line_chart.

    uploadData is contractually the output of upload_init() — a dict with
    canonical keys for the chosen domain mapped to 1-D float ndarrays. Tests
    here exercise the user-fixable validation paths (ValueError → HTTP 400
    via the Flask route) and the contract assertions (AssertionError → HTTP
    500 — only reachable via a server bug).
    """

    @staticmethod
    def _freq(f, s, l):
        return {'Frequency': np.asarray(f, float),
                'E Storage': np.asarray(s, float),
                'E Loss':    np.asarray(l, float)}

    @staticmethod
    def _temp(T, s, l):
        return {'Temperature': np.asarray(T, float),
                'E Storage':   np.asarray(s, float),
                'E Loss':      np.asarray(l, float)}

    def _call(self, data, **kw):
        return update_line_chart(
            data, number_of_prony=5, smoothness=0.1, fit_settings=True, **kw,
        )

    def test_empty_data_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'no data rows'):
            self._call(self._freq([], [], []), domain='frequency')

    def test_nan_in_data_raises_value_error(self):
        bad = self._freq([1.0, 2.0, 3.0], [100.0, np.nan, 300.0], [10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self._call(bad, domain='frequency')

    def test_inf_in_data_raises_value_error(self):
        bad = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, np.inf, 30.0])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self._call(bad, domain='frequency')

    def test_zero_frequency_raises_value_error(self):
        bad = self._freq([0.0, 1.0, 10.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, 'Frequency.*positive'):
            self._call(bad, domain='frequency')

    def test_negative_frequency_raises_value_error(self):
        bad = self._freq([-1.0, 1.0, 10.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, 'Frequency.*positive'):
            self._call(bad, domain='frequency')

    # --- optional error columns must be strictly positive -------------------
    # They divide the residuals as 1/sigma weights, so a zero blows up the
    # weighted design matrix and a negative is silently meaningless.

    def test_zero_shared_error_column_raises_value_error(self):
        bad = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['Error'] = np.asarray([1.0, 0.0, 3.0], float)
        with self.assertRaisesRegex(ValueError, "'Error'.*positive"):
            self._call(bad, domain='frequency')

    def test_negative_shared_error_column_raises_value_error(self):
        # Accepted silently before this check: the squared residual is
        # sign-invariant, so a negative sigma fits without complaint.
        bad = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['Error'] = np.asarray([1.0, -2.0, 3.0], float)
        with self.assertRaisesRegex(ValueError, "'Error'.*positive"):
            self._call(bad, domain='frequency')

    def test_zero_storage_error_column_names_that_column(self):
        bad = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['E Storage Error'] = np.asarray([5.0, 0.0, 15.0], float)
        bad['E Loss Error'] = np.asarray([0.5, 1.0, 1.5], float)
        with self.assertRaisesRegex(ValueError, "'E Storage Error'.*positive"):
            self._call(bad, domain='frequency')

    def test_zero_loss_error_column_names_that_column(self):
        # Mirrors the above so the loop can't be shown to short-circuit on the
        # first error column it inspects.
        bad = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['E Storage Error'] = np.asarray([5.0, 10.0, 15.0], float)
        bad['E Loss Error'] = np.asarray([0.5, 0.0, 1.5], float)
        with self.assertRaisesRegex(ValueError, "'E Loss Error'.*positive"):
            self._call(bad, domain='frequency')

    def test_positive_error_columns_pass_validation(self):
        # Negative control: the new check must not over-reject valid files.
        ok = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        ok['E Storage Error'] = np.asarray([5.0, 10.0, 15.0], float)
        ok['E Loss Error'] = np.asarray([0.5, 1.0, 1.5], float)
        self.assertEqual(len(self._call(ok, domain='frequency')), 10)

    def test_temperature_domain_zero_error_raises_value_error(self):
        # WLF params chosen as in TestUpdateLineChartErrorColumns so every row
        # stays inside the valid shift window.
        bad = self._temp([10.0, 25.0, 40.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['Error'] = np.asarray([1.0, 0.0, 3.0], float)
        with self.assertRaisesRegex(ValueError, "'Error'.*positive"):
            self._call(bad, domain='temperature',
                       Tg=25.0, C1=17.44, C2=51.6, shift_model='WLF')

    def test_temperature_domain_zero_error_raises_before_shift_param_shortcut(self):
        # With no shift params the temperature branch returns placeholder
        # figures early, never reaching the fit. The check must still fire, which
        # pins it ahead of that shortcut rather than beside the sigma lookup.
        bad = self._temp([10.0, 25.0, 40.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad['Error'] = np.asarray([1.0, 0.0, 3.0], float)
        with self.assertRaisesRegex(ValueError, "'Error'.*positive"):
            self._call(bad, domain='temperature')

    def test_wrong_uploadData_keys_raises_assertion(self):
        # Server-bug class: uploadData doesn't match upload_init's contract.
        bad = {'Frequency': np.array([1.0, 2.0, 3.0])}
        with self.assertRaises(AssertionError):
            self._call(bad, domain='frequency')

    def test_unknown_domain_raises_assertion(self):
        # Server-bug class: frontend only ever sends 'frequency' or 'temperature'.
        ok = self._freq([1.0, 2.0, 3.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        with self.assertRaises(AssertionError):
            self._call(ok, domain='bogus')

    def test_shiftdata_positional_length_mismatch_raises_value_error(self):
        # Legacy positional shiftData (no Temperature column) still requires the
        # row count to match. With a Temperature column it would instead
        # interpolate onto the data temperatures (see test_shift_factors).
        temp = self._temp([0.0, 25.0, 50.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad_shift = {'a_T': [1.0, 1.0]}
        with self.assertRaisesRegex(ValueError, 'row'):
            self._call(temp, domain='temperature', shiftData=bad_shift)

    def test_shiftdata_missing_a_T_raises_assertion(self):
        # Server-bug class: upload_init(..., 'shift') always produces an 'a_T' key,
        # so a missing one means someone constructed shiftData by hand.
        temp = self._temp([0.0, 25.0, 50.0], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
        bad_shift = {'Temperature': [0.0, 25.0, 50.0], 'wrong_key': [1.0, 1.0, 1.0]}
        with self.assertRaises(AssertionError):
            self._call(temp, domain='temperature', shiftData=bad_shift)


class TestUpdateLineChartErrorColumns(unittest.TestCase):
    """update_line_chart should pass E_stor_std/E_loss_std from the uploadData
    error columns when present, else fall back to relative_error*|E*|."""

    @staticmethod
    def _freq_data(extras=None):
        d = {
            'Frequency': np.array([1.0, 2.0, 3.0]),
            'E Storage': np.array([100.0, 200.0, 300.0]),
            'E Loss':    np.array([10.0, 20.0, 30.0]),
        }
        if extras:
            d.update(extras)
        return d

    @staticmethod
    def _temp_data(extras=None):
        # Temperatures stay within the WLF valid window for Tg=25, C2=51.6
        # (|log10 a_T| < MAX_ABS_LOG10_SHIFT) so tts_temperature_to_frequency_V2
        # keeps all three rows; the post-shift Frequency sort still reverses
        # their order, which is what the flow-through test exercises.
        d = {
            'Temperature': np.array([10.0, 25.0, 40.0]),
            'E Storage':   np.array([100.0, 200.0, 300.0]),
            'E Loss':      np.array([10.0, 20.0, 30.0]),
        }
        if extras:
            d.update(extras)
        return d

    def _spy(self):
        # Return a minimal valid fit result so downstream figure builders run.
        # update_line_chart asks for return_fit_quality, hence the third element.
        spy_target = patch(
            'app.trive.chart.smooth_prony_fit',
            return_value=(np.array([1.0]), np.array([1.0]),
                          _FitQuality(1.0, 2.0, 3.0)),
        )
        return spy_target

    def test_per_modulus_error_columns_flow_into_smooth_prony_fit(self):
        stor_err = np.array([5.0, 10.0, 15.0])
        loss_err = np.array([0.5, 1.0, 1.5])
        data = self._freq_data({
            'E Storage Error': stor_err,
            'E Loss Error':    loss_err,
        })
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency',
            )
        kwargs = spy.call_args.kwargs
        np.testing.assert_array_equal(kwargs['E_stor_std'], stor_err)
        np.testing.assert_array_equal(kwargs['E_loss_std'], loss_err)

    def test_shared_error_column_flows_into_both_std_kwargs(self):
        err = np.array([1.0, 2.0, 3.0])
        data = self._freq_data({'Error': err})
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency',
            )
        kwargs = spy.call_args.kwargs
        np.testing.assert_array_equal(kwargs['E_stor_std'], err)
        np.testing.assert_array_equal(kwargs['E_loss_std'], err)

    def test_no_error_columns_falls_back_to_default_relative_error(self):
        data = self._freq_data()
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency',
            )
        kwargs = spy.call_args.kwargs
        expected = np.abs(data['E Storage'] + 1.0j * data['E Loss']) * 0.2
        # The factor rides in std_scale rather than the array, so the sigma the
        # fit actually sees is the product. Asserting the product keeps this
        # test about the weighting rather than about where the factor is held.
        np.testing.assert_allclose(
            kwargs['E_stor_std'] * kwargs['std_scale'], expected)
        np.testing.assert_allclose(
            kwargs['E_loss_std'] * kwargs['std_scale'], expected)

    def test_relative_error_kwarg_scales_fallback(self):
        data = self._freq_data()
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency',
                relative_error=0.5,
            )
        kwargs = spy.call_args.kwargs
        expected = np.abs(data['E Storage'] + 1.0j * data['E Loss']) * 0.5
        np.testing.assert_allclose(
            kwargs['E_stor_std'] * kwargs['std_scale'], expected)
        np.testing.assert_allclose(
            kwargs['E_loss_std'] * kwargs['std_scale'], expected)
        # The array itself must stay free of the factor — that is what lets two
        # relative-error moves share a reduction.
        np.testing.assert_allclose(
            kwargs['E_stor_std'],
            np.abs(data['E Storage'] + 1.0j * data['E Loss']))

    def test_error_columns_leave_std_scale_at_one(self):
        # relative_error is meaningless when the file supplies sigma, so the
        # factor must not leak into std_scale and rescale the user's own errors.
        data = self._freq_data({'Error': np.array([1.0, 2.0, 3.0])})
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency', relative_error=0.5,
            )
        self.assertEqual(spy.call_args.kwargs['std_scale'], 1.0)

    def test_error_scale_kwarg_rides_in_std_scale_with_columns(self):
        # The scale must ride in std_scale rather than the array, so every
        # move of the Error Scale widget shares one cached reduction — the
        # same trick relative_error uses on the no-columns path.
        err = np.array([1.0, 2.0, 3.0])
        data = self._freq_data({'Error': err})
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency', error_scale=2.0,
            )
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs['std_scale'], 2.0)
        np.testing.assert_array_equal(kwargs['E_stor_std'], err)
        np.testing.assert_array_equal(kwargs['E_loss_std'], err)

    def test_error_scale_ignored_without_columns(self):
        # The mirror of test_error_columns_leave_std_scale_at_one: with no
        # columns the widget is in Relative Error mode, and the scale value
        # (sent on every request regardless) must not touch the fallback.
        data = self._freq_data()
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='frequency',
                relative_error=0.5, error_scale=3.0,
            )
        self.assertEqual(spy.call_args.kwargs['std_scale'], 0.5)

    def test_temperature_per_modulus_error_columns_flow_through_tts(self):
        # The temperature branch routes data through tts_temperature_to_frequency_V2,
        # which reorders rows by post-shift Frequency. The per-row error values
        # must ride along that reorder so smooth_prony_fit sees them aligned.
        stor_err = np.array([5.0, 10.0, 15.0])
        loss_err = np.array([0.5, 1.0, 1.5])
        data = self._temp_data({
            'E Storage Error': stor_err,
            'E Loss Error':    loss_err,
        })
        with self._spy() as spy:
            update_line_chart(
                data, number_of_prony=5, smoothness=0.1,
                fit_settings=False, domain='temperature',
                Tg=25.0, C1=17.44, C2=51.6, shift_model='WLF',
            )
        kwargs = spy.call_args.kwargs
        # Each (storage, loss) error pair must still be co-located with its
        # source row after the frequency-sort reorder. The set of pairs is
        # therefore invariant under TTS even though the order changes.
        pairs_out = set(zip(kwargs['E_stor_std'].tolist(),
                            kwargs['E_loss_std'].tolist()))
        pairs_in = set(zip(stor_err.tolist(), loss_err.tolist()))
        self.assertEqual(pairs_out, pairs_in)


class TestUpdateLineChartPlotDecimation(unittest.TestCase):
    """Plot-trace thinning for oversized uploads: figures shrink and carry a
    notice; the fit and coefficient table still use every row."""

    N_LARGE = 5001  # > _PLOT_MAX_POINTS, indivisible spacing

    @staticmethod
    def _frequency_upload(n_rows):
        # Peaked Prony source sampled densely — a miniature of the broadband
        # chirp master curves that motivated the thinning.
        tau = np.logspace(-4.0, 4.0, 9)
        E_input = np.concatenate(
            ([1e6], np.exp(-(np.log10(tau)) ** 2 / 4.0) * 1e9))
        df = compute_complex(tau, E_input, num_pts=n_rows)
        return {
            'Frequency': df['Frequency'].to_numpy(),
            'E Storage': df['E Storage'].to_numpy(),
            'E Loss': df['E Loss'].to_numpy(),
        }

    @staticmethod
    def _experiment_lengths(fig):
        return [len(t.x) for t in fig.data if t.name == 'Experiment']

    @staticmethod
    def _decimation_notices(fig):
        return [a.text for a in fig.layout.annotations
                if a.text and 'decimated by' in a.text]

    @staticmethod
    def _quality_readouts(fig):
        return [a.text for a in fig.layout.annotations
                if a.text and 'lower is better' in a.text]

    @classmethod
    def setUpClass(cls):
        cls.uploadData = cls._frequency_upload(cls.N_LARGE)
        # shift_model is required for fig4/fig41 to be built at all; the
        # decimation assertions below cover them, so ask for the transform.
        cls.result = update_line_chart(
            cls.uploadData, number_of_prony=10, smoothness=0.0,
            fit_settings=False, domain='frequency',
            shift_model='WLF', Tg=30.0, C1=17.44, C2=51.6,
        )

    def test_large_upload_experiment_traces_are_thinned(self):
        fig1, fig11, _, _, fig4, fig41, _, _, _, _ = self.result
        for fig in (fig1, fig11, fig4, fig41):
            lengths = self._experiment_lengths(fig)
            self.assertTrue(lengths)  # experiment traces exist
            self.assertTrue(all(n <= _PLOT_MAX_POINTS for n in lengths))
            # thinned, not truncated: still a substantial trace
            self.assertTrue(all(n > _PLOT_MAX_POINTS // 2 for n in lengths))

    def test_prony_model_traces_untouched(self):
        # The model overlay comes from compute_complex(num_pts=1000), not from
        # the experiment rows, so thinning must not alter it.
        fig1 = self.result[0]
        model_lengths = [len(t.x) for t in fig1.data if 'Term Prony' in t.name]
        self.assertEqual(model_lengths, [1000, 1000])

    def test_figures_carry_decimation_notice_with_percentage(self):
        fig1, fig11, _, _, fig4, fig41, _, _, _, _ = self.result
        expected_pct = int(round(100.0 * (1 - _PLOT_MAX_POINTS / self.N_LARGE)))
        for fig in (fig1, fig11, fig4, fig41):
            notices = self._decimation_notices(fig)
            self.assertEqual(len(notices), 1)
            self.assertIn('too many data points', notices[0])
            self.assertIn(f'{expected_pct}%', notices[0])

    def test_complex_figures_carry_fit_quality_readout(self):
        # The two figures that overlay the fit on the data get the scores; the
        # temperature-domain visualizations, which show no fit, do not.
        fig1, fig11, _, _, fig4, fig41, _, _, _, _ = self.result
        for fig in (fig1, fig11):
            readouts = self._quality_readouts(fig)
            self.assertEqual(len(readouts), 1)
            self.assertIn('χ²/ν', readouts[0])
            self.assertIn('lower is better', readouts[0])
        for fig in (fig4, fig41):
            self.assertEqual(self._quality_readouts(fig), [])

    @staticmethod
    def _notices(fig):
        return [a for a in fig.layout.annotations
                if a.text and ('decimated by' in a.text or 'lower is better' in a.text)]

    def test_decimation_and_quality_notices_do_not_share_a_row(self):
        # Both strings are long — the decimation notice runs 86 characters and
        # the readout over 100 — so each crosses the middle of the plot on its
        # own. Anchoring one left and the other right on a single row is not
        # enough to keep them apart; they have to stack.
        fig1, fig11 = self.result[0], self.result[1]
        for fig in (fig1, fig11):
            notices = self._notices(fig)
            self.assertEqual(len(notices), 2, msg='expected both notices')
            self.assertEqual(
                len({a.y for a in notices}), 2,
                msg='decimation notice and fit-quality readout share a row',
            )

    def test_fit_quality_sits_below_the_decimation_notice(self):
        # The readout is the number a user watches while dragging the sliders,
        # so it takes the row nearest the plot and the decimation notice — which
        # says the same thing on every move — stacks above it.
        fig1 = self.result[0]
        quality = [a for a in self._notices(fig1) if 'lower is better' in a.text][0]
        decimation = [a for a in self._notices(fig1) if 'decimated by' in a.text][0]
        self.assertLess(quality.y, decimation.y)

    def test_notices_are_right_aligned(self):
        # Ragged left, flush right: the readout's numbers change width from fit
        # to fit, and anchoring right keeps that from shifting the block.
        for fig in (self.result[0], self.result[1], self.result[4]):
            for a in self._notices(fig):
                self.assertEqual(a.xanchor, 'right')
                self.assertEqual(a.x, 1.0)

    def test_lone_notice_stays_on_the_bottom_row(self):
        # The temperature figures carry the decimation notice with no readout
        # beside it. Stacking must not push a solitary note up into the margin
        # and leave an empty row under it.
        fig1, fig4 = self.result[0], self.result[4]
        lone = self._notices(fig4)
        self.assertEqual(len(lone), 1)
        self.assertEqual(lone[0].y, min(a.y for a in self._notices(fig1)))

    def test_stacking_makes_headroom_for_the_upper_row(self):
        # A second row sits higher above the plot than plotly express's default
        # 60px top margin leaves room for, so it would be clipped without more.
        # The class fixture's fig1 stacks two rows (decimation + quality
        # readout); a small upload's fig1 carries the quality readout alone.
        one_row = update_line_chart(
            self._frequency_upload(200), number_of_prony=5, smoothness=0.0,
            fit_settings=False, domain='frequency',
        )[0]
        self.assertGreater(self.result[0].layout.margin.t,
                           one_row.layout.margin.t)

    def test_quality_readout_omits_posterior_when_unsmoothed(self):
        # setUpClass fits with smoothness=0, so there is no posterior over the
        # smoothing weight and only the misfit should be shown. The NNLS active
        # set also leaves exact zeros, so log-spectrum roughness is undefined.
        readout = self._quality_readouts(self.result[0])[0]
        self.assertNotIn('surprisal', readout)
        self.assertNotIn('curvature', readout)

    def test_quality_readout_shows_posterior_when_smoothed(self):
        smoothed = update_line_chart(
            self.uploadData, number_of_prony=10, smoothness=1.0,
            fit_settings=False, domain='frequency',
        )
        readout = self._quality_readouts(smoothed[0])[0]
        self.assertIn('χ²/ν', readout)
        self.assertIn('surprisal', readout)
        self.assertIn('curvature', readout)

    def test_quality_readout_puts_curvature_between_the_other_two(self):
        # chi-squared and curvature are the two L-curve coordinates, so they
        # read as a pair; the posterior is a separate criterion and goes last.
        smoothed = update_line_chart(
            self.uploadData, number_of_prony=10, smoothness=1.0,
            fit_settings=False, domain='frequency',
        )
        readout = self._quality_readouts(smoothed[0])[0]
        self.assertLess(readout.index('χ²/ν'), readout.index('curvature'))
        self.assertLess(readout.index('curvature'), readout.index('surprisal'))

    def test_annotations_preserve_plotly_express_facet_titles(self):
        # add_annotation is additive; update_layout(annotations=...) would have
        # replaced the facet labels these faceted figures depend on.
        for fig in (self.result[0], self.result[1]):
            facet_titles = [a.text for a in fig.layout.annotations
                            if a.text and a.text.startswith('Modulus=')]
            self.assertTrue(facet_titles)

    def test_fit_uses_all_rows_not_the_thinned_frame(self):
        # Fitting the full arrays directly must reproduce the coefficients
        # update_line_chart returned; a fit on thinned data would differ.
        freq = self.uploadData['Frequency']
        es = self.uploadData['E Storage']
        el = self.uploadData['E Loss']
        std = np.abs(es + 1.0j * el) * 0.2  # default relative_error path
        tau_i, E_i = smooth_prony_fit(
            freq, es, el, E_stor_std=std, E_loss_std=std,
            N=10, smoothness=0.0, solid=True,
        )
        coef = {row['i']: row['E_i'] for row in self.result[6]}
        expected = {i: e for i, e in enumerate(E_i[1:]) if e != 0}
        self.assertEqual(set(coef), set(expected))
        for i in coef:
            np.testing.assert_allclose(coef[i], expected[i], rtol=1e-10)

    def test_small_upload_untouched_and_unannotated(self):
        result = update_line_chart(
            self._frequency_upload(200), number_of_prony=5, smoothness=0.0,
            fit_settings=False, domain='frequency',
            shift_model='WLF', Tg=30.0, C1=17.44, C2=51.6,
        )
        fig1, _, _, _, fig4, _, _, _, _, _ = result
        self.assertTrue(all(n == 200 for n in self._experiment_lengths(fig1)))
        for fig in (fig1, fig4):
            self.assertEqual(self._decimation_notices(fig), [])

    def test_temperature_domain_figures_also_thinned(self):
        # Temperature branch without shift params: early return, only the
        # temperature figures are built — they must still thin and annotate.
        n = self.N_LARGE
        data = {
            'Temperature': np.linspace(-50.0, 150.0, n),
            'E Storage': np.linspace(1e9, 1e6, n),
            'E Loss': np.full(n, 1e5),
        }
        _, _, _, _, fig4, fig41, _, _, _, _ = update_line_chart(
            data, number_of_prony=5, smoothness=0.0,
            fit_settings=False, domain='temperature',
        )
        for fig in (fig4, fig41):
            lengths = self._experiment_lengths(fig)
            self.assertTrue(lengths)
            self.assertTrue(all(n_ <= _PLOT_MAX_POINTS for n_ in lengths))
            self.assertEqual(len(self._decimation_notices(fig)), 1)


class TestUpdateLineChartRankInfo(unittest.TestCase):
    """The rank_info element and the noise-rank caption it drives."""

    @staticmethod
    def _frequency_upload(n_rows=300):
        tau = np.logspace(-3.0, 3.0, 7)
        E_input = np.concatenate(
            ([1e6], np.exp(-(np.log10(tau)) ** 2 / 4.0) * 1e9))
        df = compute_complex(tau, E_input, num_pts=n_rows)
        return {
            'Frequency': df['Frequency'].to_numpy(),
            'E Storage': df['E Storage'].to_numpy(),
            'E Loss': df['E Loss'].to_numpy(),
        }

    @staticmethod
    def _rank_notices(fig):
        return [a.text for a in fig.layout.annotations
                if a.text and 'identifiable' in a.text]

    def _run(self, **overrides):
        args = dict(
            uploadData=self._frequency_upload(),
            number_of_prony=10, smoothness=0.0,
            fit_settings=False, domain='frequency',
        )
        args.update(overrides)
        return update_line_chart(**args)

    def test_frequency_result_carries_rank_info(self):
        rank_info = self._run()[9]
        self.assertEqual(set(rank_info), {'max_prony', 'noise_prony'})
        # Plain ints (the route serializes with stdlib json.dumps), in range.
        self.assertIs(type(rank_info['max_prony']), int)
        self.assertIs(type(rank_info['noise_prony']), int)
        self.assertGreaterEqual(rank_info['max_prony'], 1)
        self.assertLessEqual(rank_info['max_prony'], 100)
        self.assertGreaterEqual(rank_info['noise_prony'], 0)

    def test_temperature_result_carries_rank_info(self):
        # With usable shift params the transform runs and the probe measures
        # the TRANSFORMED master curve — same shape of result as frequency.
        # T span clear of the WLF C2 singularity at Tg - C2, as in
        # TestUpdateLineChartTemperature.
        T = np.linspace(0.0, 80.0, 30)
        rank_info = update_line_chart(
            {
                'Temperature': T,
                'E Storage': np.linspace(1000.0, 10.0, len(T)),
                'E Loss': np.full(len(T), 50.0),
            },
            number_of_prony=8, smoothness=0.1,
            fit_settings=False, domain='temperature',
            Tg=25.0, C1=17.44, C2=51.6, shift_model='WLF',
        )[9]
        self.assertEqual(set(rank_info), {'max_prony', 'noise_prony'})
        self.assertGreaterEqual(rank_info['max_prony'], 1)

    def test_caption_when_terms_exceed_noise_rank(self):
        # An enormous stated relative error leaves almost nothing
        # identifiable, so a full-size request must be called out — on both
        # figures that overlay the fit, with the count bound embedded. The
        # wording is a BOUND ("at most ~k"), never a partition of the
        # requested N: noise_prony counts singular directions, and no
        # particular term is the identifiable one.
        result = self._run(number_of_prony=100, relative_error=1000.0)
        noise_prony = result[9]['noise_prony']
        self.assertLess(noise_prony, 100)
        for fig in (result[0], result[1]):
            notices = self._rank_notices(fig)
            self.assertEqual(len(notices), 1)
            self.assertIn('if the error profile is accurate', notices[0])
            self.assertIn(f'at most ~{noise_prony} terms', notices[0])
            self.assertNotIn('of 100', notices[0])

    def test_caption_wording_stays_out_of_the_other_filters(self):
        # The decimation and quality test helpers select notices by substring;
        # the rank caption must never match either filter.
        result = self._run(number_of_prony=100, relative_error=1000.0)
        notice = self._rank_notices(result[0])[0]
        self.assertNotIn('decimated by', notice)
        self.assertNotIn('lower is better', notice)

    def test_no_caption_when_error_profile_supports_the_request(self):
        # A tight error profile determines more modes than the request — the
        # healthy case adds no visual noise.
        result = self._run(number_of_prony=5, relative_error=0.001)
        self.assertGreaterEqual(result[9]['noise_prony'], 5)
        for fig in (result[0], result[1]):
            self.assertEqual(self._rank_notices(fig), [])

    def test_rank_caption_stacks_a_third_row_with_headroom(self):
        # Decimation notice + quality readout + rank caption is the first
        # three-row stack; the top margin must keep growing so the new top
        # row is not clipped (see _stamp_notice).
        big = self._frequency_upload(_PLOT_MAX_POINTS + 500)
        kwargs = dict(
            uploadData=big, number_of_prony=10, smoothness=1.0,
            fit_settings=False, domain='frequency',
        )
        two_rows = update_line_chart(relative_error=0.2, **kwargs)[0]
        three_rows = update_line_chart(relative_error=1000.0, **kwargs)[0]
        self.assertEqual(len(self._rank_notices(two_rows)), 0)
        self.assertEqual(len(self._rank_notices(three_rows)), 1)
        self.assertGreater(three_rows.layout.margin.t,
                           two_rows.layout.margin.t)
        # Same-thing-every-move notices stack above the readout being watched:
        # the rank caption takes the top row.
        stamped = [a for a in three_rows.layout.annotations
                   if a.name == 'figure-notice']
        self.assertEqual(len(stamped), 3)
        top = max(stamped, key=lambda a: a.y)
        self.assertIn('identifiable', top.text)


if __name__ == '__main__':
    unittest.main()
