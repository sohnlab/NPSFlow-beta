"""Detection BC multi-zone assembly + runner wrapping. Mocks the Qt dialog."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def test_merge_zone_regions_union_and_distance():
    from processing.pulse_detection_bc import _merge_zone_regions
    z0 = np.array([[10, 20], [100, 110]])
    z1 = np.array([[15, 25], [200, 210]])   # [15,25] overlaps [10,20]
    # margin=0, merge_distance=0: only overlapping/touching merge
    out = _merge_zone_regions([z0, z1], margin=0, merge_distance=0, n=1000)
    # [10,20]+[15,25] -> [10,25]; [100,110]; [200,210]
    assert out.tolist() == [[10, 25], [100, 110], [200, 210]]


def test_merge_zone_regions_distance_bridges_gap():
    from processing.pulse_detection_bc import _merge_zone_regions
    z0 = np.array([[10, 20]])
    z1 = np.array([[30, 40]])               # gap of 10 between 20 and 30
    out = _merge_zone_regions([z0, z1], margin=0, merge_distance=10, n=1000)
    assert out.tolist() == [[10, 40]]       # gap (10) <= merge_distance -> merged
    out2 = _merge_zone_regions([z0, z1], margin=0, merge_distance=5, n=1000)
    assert out2.tolist() == [[10, 20], [30, 40]]   # gap > 5 -> separate


def test_merge_zone_regions_margin_and_clamp():
    from processing.pulse_detection_bc import _merge_zone_regions
    z0 = np.array([[2, 5]])
    out = _merge_zone_regions([z0], margin=5, merge_distance=0, n=10)
    assert out.tolist() == [[0, 9]]         # 2-5 -> clamp low 0, 5+5=10 clamp to n-1=9


def test_merge_zone_regions_empty():
    from processing.pulse_detection_bc import _merge_zone_regions
    out = _merge_zone_regions([np.empty((0, 2), int), np.empty((0, 2), int)],
                              margin=0, merge_distance=0, n=100)
    assert out.shape == (0, 2)


def test_assemble_zone_results_dual_outputs_multi():
    from processing.pulse_detection_bc import _assemble_zone_results
    per_zone = [
        {"trendline": np.zeros(5), "detectedPulses": np.array([[0, 2]])},
        {"trendline": np.ones(5),  "detectedPulses": np.array([[1, 3]])},
    ]
    out = _assemble_zone_results(per_zone)
    assert out["trendline"].shape == (5, 2)
    # detectedPulses is a MERGED Nx2 (not zone-grouped). [0,2]+[1,3] overlap -> [0,3]
    assert out["detectedPulses"].tolist() == [[0, 3]]
    # detectedPerZone keeps the per-zone regions, zone-grouped
    dpz = out["detectedPerZone"]
    assert dpz["num_zones"] == 2
    assert dpz["zones"][1].tolist() == [[1, 3]]


def test_assemble_zone_results_dual_outputs_single():
    from processing.pulse_detection_bc import _assemble_zone_results
    per_zone = [{"trendline": np.zeros(5), "detectedPulses": np.array([[0, 2]])}]
    out = _assemble_zone_results(per_zone)
    assert out["trendline"].ndim == 1
    assert out["detectedPulses"].tolist() == [[0, 2]]      # bare, == the one zone
    # single zone: detectedPerZone is bare Nx2 (wrap_zones of one item)
    assert np.asarray(out["detectedPerZone"]).tolist() == [[0, 2]]


def test_dialog_zone_tab_bar_and_roundtrip():
    """Headless: single-zone keeps the bar hidden; multi-zone switches state."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * 5 * t) + 0.05 * rng.standard_normal(n)

    # --- single zone: bar hidden, plain attrs populated ---
    try:
        dlg1 = _PulseDetectionBCDialog([sig.copy()], fs)
    except Exception as e:
        raise AssertionError(f"single-zone construction failed: {e!r}")
    assert len(dlg1._zones) == 1
    # Dialog is never shown headless, so isVisible() is always False; check the
    # explicit hidden flag instead (single-zone hides the bar).
    assert dlg1._zone_bar.isHidden()
    assert dlg1._active_zone == 0
    assert dlg1._data is not None
    assert dlg1._n == n
    dlg1.close()

    # --- two zones: bar visible with 2 tabs, save/load round-trip ---
    two = np.column_stack([sig, sig * 0.5 + 0.1])
    try:
        dlg2 = _PulseDetectionBCDialog([two[:, 0], two[:, 1]], fs)
    except Exception as e:
        raise AssertionError(f"two-zone construction failed: {e!r}")
    assert not dlg2._zone_bar.isHidden()  # multi-zone shows the bar
    # Two real zones + the appended "All Zones" tab.
    assert dlg2._zone_bar.count() == 3
    assert len(dlg2._zones) == 2

    sentinel = np.array([[10, 20]])
    dlg2._pulse_regions = sentinel.copy()
    dlg2._on_zone_changed(1)
    assert dlg2._active_zone == 1
    dlg2._on_zone_changed(0)
    assert dlg2._active_zone == 0
    assert np.array_equal(dlg2._pulse_regions, sentinel), \
        "pulse_regions sentinel did not survive the zone round-trip"
    dlg2.close()


def test_per_zone_cutoff_isolation():
    """Each zone keeps its own derivative cutoff across switches (#2/#3)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 5 * t)
    dlg = _PulseDetectionBCDialog([sig, sig * 0.5], fs)

    # Zone 0: set a distinct cutoff via the spinbox + Apply.
    dlg._spin_cutoff.setValue(60.0)
    dlg._on_apply_detection()
    assert abs(dlg._applied_cutoff - 60.0) < 1e-6

    # Switch to zone 1, give it a different cutoff.
    dlg._on_zone_changed(1)
    dlg._spin_cutoff.setValue(150.0)
    dlg._on_apply_detection()
    assert abs(dlg._applied_cutoff - 150.0) < 1e-6

    # Back to zone 0: must restore 60, not bleed 150 from zone 1.
    dlg._on_zone_changed(0)
    assert abs(dlg._applied_cutoff - 60.0) < 1e-6, \
        f"zone 0 cutoff bled: got {dlg._applied_cutoff}"
    assert abs(dlg._zones[0].applied_cutoff - 60.0) < 1e-6
    assert abs(dlg._zones[1].applied_cutoff - 150.0) < 1e-6
    dlg.close()


def test_per_zone_merge_factor_isolation():
    """Region Margin / Merge Distance are per-zone; editing one zone's merge
    factors must not bleed into another zone."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 5 * t)
    dlg = _PulseDetectionBCDialog([sig, sig * 0.5], fs)

    # Zone 0: distinct merge factors via the spinboxes.
    dlg._spin_padding.setValue(0.30)
    dlg._spin_spacing.setValue(0.70)
    dlg._on_apply_detection()

    # Switch to zone 1: a fresh zone must start from defaults, not bleed 0.30/0.70.
    dlg._on_zone_changed(1)
    assert abs(dlg._spin_padding.value() - 0.20) < 1e-6, \
        f"zone 1 margin bled from zone 0: {dlg._spin_padding.value()}"
    assert abs(dlg._spin_spacing.value() - 0.50) < 1e-6, \
        f"zone 1 distance bled from zone 0: {dlg._spin_spacing.value()}"
    dlg._spin_padding.setValue(0.45)
    dlg._spin_spacing.setValue(1.20)
    dlg._on_apply_detection()

    # Back to zone 0: must restore 0.30/0.70, not zone 1's 0.45/1.20.
    dlg._on_zone_changed(0)
    assert abs(dlg._spin_padding.value() - 0.30) < 1e-6, \
        f"zone 0 margin bled: got {dlg._spin_padding.value()}"
    assert abs(dlg._spin_spacing.value() - 0.70) < 1e-6, \
        f"zone 0 distance bled: got {dlg._spin_spacing.value()}"
    assert abs(dlg._zones[0].region_margin - 0.30) < 1e-6
    assert abs(dlg._zones[0].merge_distance - 0.70) < 1e-6
    assert abs(dlg._zones[1].region_margin - 0.45) < 1e-6
    assert abs(dlg._zones[1].merge_distance - 1.20) < 1e-6
    dlg.close()


def test_per_zone_merge_factor_save_restore():
    """Per-zone merge factors round-trip through get_settings / load_settings."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    rng = np.random.default_rng(0)
    sig0 = rng.standard_normal(2000)
    sig1 = rng.standard_normal(2000)

    dlg1 = _PulseDetectionBCDialog([sig0, sig1], fs)
    dlg1._spin_padding.setValue(0.33)
    dlg1._spin_spacing.setValue(0.66)
    dlg1._zones[1].region_margin = 0.44
    dlg1._zones[1].merge_distance = 1.10
    s = dlg1.get_settings()
    dlg1.close()

    assert abs(s["zones"][0]["region_margin"] - 0.33) < 1e-6
    assert abs(s["zones"][0]["merge_distance"] - 0.66) < 1e-6
    assert abs(s["zones"][1]["region_margin"] - 0.44) < 1e-6
    assert abs(s["zones"][1]["merge_distance"] - 1.10) < 1e-6

    dlg2 = _PulseDetectionBCDialog([sig0, sig1], fs)
    dlg2.load_settings(s)
    assert abs(dlg2._zones[0].region_margin - 0.33) < 1e-6
    assert abs(dlg2._zones[1].region_margin - 0.44) < 1e-6
    assert abs(dlg2._zones[1].merge_distance - 1.10) < 1e-6
    dlg2.close()


def test_auto_tangent_default_linear():
    """Fresh single zone defaults to linear baseline + Linear radio (#1)."""
    from processing.pulse_detection_bc import _PerZoneState
    assert _PerZoneState(np.zeros(10)).use_auto_tangent is False

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 5 * t)
    dlg = _PulseDetectionBCDialog([sig], fs)
    assert dlg._use_auto_tangent is False
    dlg._build_baseline_panel()
    assert dlg._radio_linear.isChecked()
    assert not dlg._radio_auto_tangent.isChecked()
    assert dlg._use_auto_tangent is False
    dlg.close()


def test_runner_passes_through_zone_output():
    """Runner returns bare output for 1 zone and zone-wrapped for 2 zones.

    Patch target: processing.pulse_detection_bc.pulse_detection_bc (module attr).
    The runner does `from processing.pulse_detection_bc import pulse_detection_bc`
    inside run(), so each call re-executes the import and picks up the patched attr.
    """
    import types
    import processing.pulse_detection_bc as real_mod

    saved = real_mod.pulse_detection_bc

    def fake_pdbc(data, *a, **kw):
        from utils.zones import zone_columns
        cols = zone_columns(np.asarray(data))
        per = [{"trendline": np.zeros(len(c)),
                "detectedPulses": np.array([[0, 1]])} for c in cols]
        return real_mod._assemble_zone_results(per)

    real_mod.pulse_detection_bc = fake_pdbc
    try:
        from runners.runPulseDetectionBC import run

        # Single zone: bare 1-D trendline, plain Nx2 detectedPulses.
        block = types.SimpleNamespace(parameters={})
        out1 = run({"dataIn": np.zeros(10), "sampleRate": 1}, {}, block)
        assert out1["trendline"].ndim == 1, \
            f"expected 1-D trendline for single zone, got shape {out1['trendline'].shape}"
        assert np.asarray(out1["detectedPulses"]).shape == (1, 2), \
            f"expected (1,2) detectedPulses for single zone, got {np.asarray(out1['detectedPulses']).shape}"

        # Two zones: 2-D trendline (N×Z), merged Nx2 detectedPulses, zone-dict detectedPerZone.
        out2 = run({"dataIn": np.zeros((10, 2)), "sampleRate": 1}, {}, block)
        assert out2["trendline"].shape == (10, 2), \
            f"expected (10,2) trendline for two zones, got {out2['trendline'].shape}"
        assert np.asarray(out2["detectedPulses"]).ndim == 2 and not isinstance(out2["detectedPulses"], dict), \
            f"expected merged Nx2 detectedPulses, got {out2['detectedPulses']}"
        assert out2["detectedPerZone"]["num_zones"] == 2, \
            f"expected num_zones=2, got {out2['detectedPerZone'].get('num_zones')}"
    finally:
        real_mod.pulse_detection_bc = saved


def test_wizard_stages_and_all_zones_merge():
    """Wizard-only stages, status zone tabs, All-Zones merge tab + output."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog

    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * 5 * t) + 0.05 * rng.standard_normal(n)

    # --- ONE zone: zone bar hidden, stage tab bar hidden, no All-Zones tab ---
    dlg1 = _PulseDetectionBCDialog([sig.copy()], fs)
    assert dlg1._zone_bar.isHidden()
    assert dlg1._tabs.tabBar().isHidden()
    assert dlg1._all_zones_idx is None
    assert dlg1._zone_bar.count() == 1
    dlg1.close()

    # --- TWO zones: bar visible, 3 tabs, All-Zones initially disabled ---
    dlg = _PulseDetectionBCDialog([sig.copy(), sig.copy() * 0.5], fs)
    assert not dlg._zone_bar.isHidden()
    assert dlg._zone_bar.count() == 3
    assert dlg._zone_bar.tabText(0).startswith("Zone 1")
    assert dlg._zone_bar.tabText(1).startswith("Zone 2")
    assert dlg._zone_bar.tabText(2) == "All Zones"
    assert dlg._all_zones_idx == 2
    assert not dlg._zone_bar.isTabEnabled(dlg._all_zones_idx)

    # All-Zones page has its own navigation (mpl toolbar + nav buttons).
    assert hasattr(dlg, '_all_zones_toolbar')
    assert hasattr(dlg, '_az_btn_zoom') and hasattr(dlg, '_az_btn_pan')

    # Mark both zones finished with regions + baseline/detrended overlay data.
    dlg._zones[0].finished = True
    dlg._zones[0].pulse_regions = np.array([[100, 150]])
    dlg._zones[0].baseline_smooth = np.zeros(n)
    dlg._zones[0].detrended_data = sig.copy()
    dlg._zones[1].finished = True
    dlg._zones[1].pulse_regions = np.array([[120, 170], [800, 820]])
    dlg._zones[1].baseline_smooth = np.zeros(n)
    dlg._zones[1].detrended_data = sig.copy() * 0.5

    dlg._update_all_zones_enabled()
    assert dlg._zone_bar.isTabEnabled(dlg._all_zones_idx)

    # Switch to the All-Zones page. (Switching snapshots the active zone, so
    # re-assert the intended per-zone regions afterward — the live working set
    # of zone 0 would otherwise overwrite our injected regions.)
    dlg._on_zone_changed(dlg._all_zones_idx)
    assert dlg._active_zone == dlg._all_zones_idx
    dlg._zones[0].pulse_regions = np.array([[100, 150]])
    dlg._zones[1].pulse_regions = np.array([[120, 170], [800, 820]])

    # Margin small; distance large enough to bridge [100,150] & [120,170].
    # template width default = 50 (no template passed). Want margin~0 samples,
    # distance >= (120-150 gap is overlap anyway). Pick factors so the merge
    # joins the overlapping pair but leaves [800,820] separate.
    dlg._spin_merge_margin.setValue(0.0)
    dlg._spin_merge_distance.setValue(1.0)   # 1.0*50 = 50 samples
    dlg._recompute_all_zones_merge()

    merged = dlg._final_merged
    assert isinstance(merged, np.ndarray) and merged.ndim == 2 and merged.shape[1] == 2
    # [100,150] and [120,170] overlap -> one region; [800,820] separate.
    assert len(merged) == 2, f"expected 2 merged regions, got {merged.tolist()}"
    assert merged[0].tolist() == [100, 170]
    assert merged[1].tolist() == [800, 820]

    # Zone 1 is drawn at the TOP (highest y-tick); zones descend.
    ticks = list(dlg._all_zones_ax.get_yticks())
    assert ticks[0] > ticks[1], f"expected descending offsets, got {ticks}"

    # Plot Height spinbox exists; changing it resizes the canvas safely.
    assert hasattr(dlg, '_spin_az_height')
    dlg._spin_az_height.setValue(300)
    assert dlg._all_zones_canvas.height() == 300 or \
        dlg._all_zones_canvas.maximumHeight() == 300

    # Activate a nav tool on All-Zones, then switch back to a real zone:
    # the All-Zones mode + its buttons must reset. (A real click toggles the
    # checkable button before the slot fires; mirror that here.)
    dlg._az_btn_pan.setChecked(True)
    dlg._on_az_toggle_pan(True)
    assert dlg._az_btn_pan.isChecked()
    assert dlg._all_zones_toolbar.mode.name == 'PAN'
    dlg._on_zone_changed(0)
    assert not dlg._az_btn_pan.isChecked()
    assert not dlg._az_btn_zoom.isChecked()
    assert dlg._all_zones_toolbar.mode.name == 'NONE'
    dlg.close()


def test_finish_zone_defers_advance_to_next_zone():
    """'Finish Zone' (multi-zone) advances to the next unfinished zone.

    The advance is deferred via QTimer.singleShot(0) so the panel rebuild
    doesn't delete the in-flight button; needs an event-loop tick to commit.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog

    fs = 1000
    n = 2000
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 5 * t)
    dlg = _PulseDetectionBCDialog([sig.copy(), sig.copy() * 0.5], fs)
    assert dlg._active_zone == 0

    # Move zone 0 into the Baseline stage so a baseline exists, then finish it.
    dlg._on_next_detection()
    assert not dlg.confirmed       # Finish Zone must NOT close the dialog
    dlg._on_finish()
    assert not dlg.confirmed
    assert dlg._zones[0].finished
    assert dlg._zones[0].trendline is not None  # captured a real baseline
    assert dlg._zone_bar.tabText(0) == "Zone 1 ✓"

    # Deferred: not advanced yet until the event loop ticks.
    app.processEvents()
    assert dlg._active_zone == 1, \
        f"expected advance to zone 1 after processEvents, got {dlg._active_zone}"
    assert dlg._zone_bar.currentIndex() == 1
    dlg.close()


def test_zone_switch_survives_deleted_detection_spinbox():
    """Switching zones must not abort when the detection-panel spinboxes have
    been destroyed (as happens in the live GUI on the Baseline stage).

    Regression for: RuntimeError 'Internal C++ object (QDoubleSpinBox) already
    deleted' in _load_active_zone, which aborted _on_zone_changed so the tab
    never switched from the Baseline stage. Headless never deletes the C++
    object on its own, so force it via shiboken6.delete.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import shiboken6
    app = QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog

    fs = 1000
    t = np.arange(4000) / fs
    sig = np.sin(2 * np.pi * 5 * t)
    dlg = _PulseDetectionBCDialog([sig.copy(), sig.copy() * 0.5], fs)

    # Simulate the live-GUI state: detection spinboxes torn down.
    assert hasattr(dlg, "_spin_cutoff")
    shiboken6.delete(dlg._spin_cutoff)
    shiboken6.delete(dlg._spin_display_cutoff)

    # Must complete the switch (previously raised RuntimeError and aborted).
    dlg._on_zone_changed(1)
    assert dlg._active_zone == 1, "zone switch aborted on deleted spinbox"
    # The rebuilt detection panel re-created live spinboxes.
    assert shiboken6.isValid(dlg._spin_cutoff)
    dlg.close()


def test_save_and_restore_all_zones():
    """get_settings serializes every zone; load_settings restores every zone
    (not just zone 0). The reported bug: only the first zone was saved."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    rng = np.random.default_rng(0)
    sig0 = rng.standard_normal(2000)
    sig1 = rng.standard_normal(2000)

    dlg1 = _PulseDetectionBCDialog([sig0, sig1], fs)
    dlg1._pos_thresh, dlg1._neg_thresh = 5.0, -5.0  # live = zone 0
    dlg1._applied_cutoff = 80.0
    dlg1._zones[1].pos_thresh = 9.0
    dlg1._zones[1].neg_thresh = -9.0
    dlg1._zones[1].applied_cutoff = 120.0
    s = dlg1.get_settings()
    dlg1.close()

    assert len(s["zones"]) == 2
    assert s["zones"][0]["pos_thresh"] == 5.0
    assert s["zones"][1]["pos_thresh"] == 9.0
    assert s["zones"][1]["deriv_cutoff"] == 120.0

    dlg2 = _PulseDetectionBCDialog([sig0, sig1], fs)
    dlg2.load_settings(s)
    assert dlg2._pos_thresh == 5.0                       # zone 0 applied live
    assert dlg2._zones[1].pos_thresh == 9.0
    assert dlg2._zones[1].restored is True
    # Visiting zone 1 applies its restored thresholds (not re-init defaults).
    dlg2._save_active_zone()
    dlg2._load_active_zone(1)
    assert dlg2._pos_thresh == 9.0
    assert dlg2._neg_thresh == -9.0
    assert dlg2._applied_cutoff == 120.0
    assert dlg2._zones[1].restored is False
    dlg2.close()

    # Legacy flat settings (no "zones") still restore the active zone.
    dlg3 = _PulseDetectionBCDialog([sig0, sig1], fs)
    dlg3.load_settings({"pos_thresh": 3.3, "neg_thresh": -3.3, "deriv_cutoff": 70.0})
    assert dlg3._pos_thresh == 3.3
    dlg3.close()


def test_get_settings_on_all_zones_page_does_not_raise():
    """The multi-zone Finish button (_on_all_zones_finish) calls get_settings
    while the All-Zones tab is active, where _active_zone == _all_zones_idx
    (out of range for _zones). get_settings must not IndexError there —
    otherwise the Finish button silently does nothing."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    fs = 1000
    rng = np.random.default_rng(0)
    dlg = _PulseDetectionBCDialog(
        [rng.standard_normal(2000), rng.standard_normal(2000)], fs)

    # Simulate standing on the All-Zones page (what the Finish handler sees).
    dlg._active_zone = dlg._all_zones_idx
    assert dlg._active_zone == len(dlg._zones)   # out of range for _zones

    s = dlg.get_settings()                       # must not raise
    assert len(s["zones"]) == 2

    # Exercise the actual finish sequence sans modal exec.
    dlg._recompute_all_zones_merge()
    dlg.confirmed = True
    dlg._cached_settings = dlg.get_settings()
    dlg._autosave()
    assert dlg.confirmed is True
    dlg.close()


def test_detection_back_navigates_zones_not_cancel():
    """Detection-stage Back on a non-first zone steps to the PREVIOUS zone
    (preserving work); on the first/single zone it cancels. Previously it
    always cancelled the whole dialog (lost every zone)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    rng = np.random.default_rng(0)
    dlg = _PulseDetectionBCDialog([rng.standard_normal(2000) for _ in range(3)], 1000)

    dlg._on_zone_changed(2)
    assert dlg._active_zone == 2
    dlg._on_detection_back()
    assert dlg._active_zone == 1          # stepped back, not cancelled
    assert dlg.result() == 0              # dialog still open
    dlg._on_detection_back()
    assert dlg._active_zone == 0
    dlg.close()

    single = _PulseDetectionBCDialog([rng.standard_normal(2000)], 1000)
    assert single._multi_zone is False
    single._on_detection_back()           # cancels (reject) — unchanged
    single.close()


def test_full_wizard_completes_and_saves_all_zones():
    """Drive the multi-zone wizard end-to-end: finishing every zone then the
    All-Zones page must not crash and must save each zone's distinct state."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from processing.pulse_detection_bc import _PulseDetectionBCDialog
    rng = np.random.default_rng(0)
    dlg = _PulseDetectionBCDialog([rng.standard_normal(2000) for _ in range(3)], 1000)
    dlg._autosave = lambda: None          # don't clobber the real temp.json

    for thr in (5.0, 7.0, 9.0):
        dlg._pos_thresh, dlg._neg_thresh = thr, -thr
        dlg._pulse_regions = np.array([[10, 20]])
        dlg._baseline_smooth = np.zeros(dlg._n)
        dlg._on_finish()                  # mark zone done + deferred advance
        app.processEvents()               # fire the QTimer zone switch (no crash)

    assert all(z.finished for z in dlg._zones)
    assert dlg._active_zone == dlg._all_zones_idx
    dlg._on_all_zones_finish()
    assert dlg.confirmed is True
    got = [z["pos_thresh"] for z in dlg._cached_settings["zones"]]
    assert got == [5.0, 7.0, 9.0], got
    dlg.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("ok ", fn.__name__)
        except Exception as e:
            failed += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
