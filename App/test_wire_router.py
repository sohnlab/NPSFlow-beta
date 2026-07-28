"""Self-running unit tests for wire_router (no Qt). Run:
    python App/test_wire_router.py
Also works under pytest if installed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import wire_router as wr


def test_segment_intersects_rect_basic():
    rect = (0, 10, 0, 10)  # x0,x1,y0,y1
    assert wr.segment_intersects_rect((-5, 5), (15, 5), rect) is True   # straight through
    assert wr.segment_intersects_rect((-5, 20), (15, 20), rect) is False  # above
    assert wr.segment_intersects_rect((2, 2), (8, 8), rect) is True      # endpoint inside
    assert wr.segment_intersects_rect((-5, -5), (-1, -1), rect) is False # fully outside


def test_inside_any():
    rects = [(0, 10, 0, 10), (20, 30, 0, 10)]
    assert wr._inside_any((5, 5), rects) is True
    assert wr._inside_any((25, 5), rects) is True
    assert wr._inside_any((15, 5), rects) is False


def test_route_clear_path_no_waypoints():
    # No obstacles -> straight, no waypoints.
    assert wr.route_wire((0, 0), (100, 0), []) == []


def test_route_around_single_block_over_or_under():
    # A block centered on the straight line; expect a 2-waypoint detour.
    block = (40, 60, -10, 10)  # x0,x1,y0,y1 (raw, unexpanded)
    wps = wr.route_wire((0, 0), (100, 0), [block])
    assert len(wps) == 2
    # Both waypoints clear the EXPANDED rect vertically (over or under).
    ex0, ex1, ey0, ey1 = wr._expand(block, wr.MARGIN)
    over = all(y >= ey1 for _, y in wps)
    under = all(y <= ey0 for _, y in wps)
    assert over or under


def test_route_excludes_far_blocks():
    # Block nowhere near the line -> no detour.
    block = (40, 60, 500, 520)
    assert wr.route_wire((0, 0), (100, 0), [block]) == []


def test_route_caps_waypoints():
    # Many stacked blocks; never exceed MAX_WAYPOINTS.
    blocks = [(10 + i * 12, 18 + i * 12, -10, 10) for i in range(20)]
    wps = wr.route_wire((0, 0), (300, 0), blocks)
    assert len(wps) <= wr.MAX_WAYPOINTS


def test_spline_clearance_after_routing():
    block = (40, 60, -10, 10)
    wps = wr.route_wire((0, 0), (100, 0), [block])
    # The rendered spline must clear the RAW block rect.
    assert wr.spline_clearance_ok((0, 0), (100, 0), wps, [block]) is True


def test_clamp_flattens_steep_boundary_crossing():
    # No blocks; a steep wire crosses an annotation's left edge at x=50.
    ann = (50, 150, -100, 100)   # x0,x1,y0,y1
    src, dst = (0, -80), (200, 80)   # slope ~0.8 -> ~38deg, steeper than 30
    wps = wr.route_wire(src, dst, [], [ann])
    # A clamp waypoint should appear just OUTSIDE the left edge (x < 50).
    assert any(x < 50 for x, _ in wps)


def test_clamp_yields_to_block():
    # A block sits exactly where the clamp waypoint would go -> clamp skipped.
    ann = (50, 150, -100, 100)
    block = (20, 50, -100, 100)   # covers the would-be clamp x (50 - margin)
    src, dst = (0, -80), (200, 80)
    wps = wr.route_wire(src, dst, [block], [ann])
    # No clamp waypoint inside the block's expanded rect.
    ex = wr._expand(block, wr.MARGIN)
    assert not any(wr._inside_any(p, [ex]) for p in wps if p[0] < 50)


def test_wire_route_mode_default_and_struct():
    from wire_connection import WireConnection
    w = WireConnection()
    assert w.route_mode == "auto"
    w.route_mode = "manual"
    w.waypoints = [(5, 5)]
    s = w.to_struct()
    assert s["routeMode"] == "manual"
    assert s["waypoints"] == [(5, 5)]


def test_route_under_detour_converges():
    # Regression: the source y lies inside the block's y-band and the detour
    # goes UNDER. Waypoints placed exactly on the rect's bottom edge used to be
    # re-detected as collisions and hit the MAX_WAYPOINTS cap. Expect a clean
    # 2-waypoint detour that clears the raw rect.
    block = (180, 340, -27, 30)   # raw block rect
    wps = wr.route_wire((160, -32), (400, -32), [block])
    assert len(wps) == 2, wps
    assert wr.spline_clearance_ok((160, -32), (400, -32), wps, [block])


def test_route_near_endpoint_obstacle_no_garbage():
    # A third block hugging the source (its expanded rect reaches back to src)
    # can't be cleanly routed around — waypoints would fall behind the endpoint
    # and zig-zag to the MAX_WAYPOINTS cap. The router drops such an obstacle and
    # accepts the short crossing instead of producing garbage.
    src, dst = (0, 0), (200, 0)
    engulfing = (14, 100, -10, 10)        # expanded left edge lands at src.x
    assert wr.route_wire(src, dst, [engulfing]) == []
    # A block just clear of the source still routes normally (2 waypoints).
    assert len(wr.route_wire(src, dst, [(16, 100, -10, 10)])) == 2
    # Symmetric case at the destination end.
    engulfing_dst = (100, 186, -10, 10)   # expanded right edge lands at dst.x
    assert wr.route_wire(src, dst, [engulfing_dst]) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
