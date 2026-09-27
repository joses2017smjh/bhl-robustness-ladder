import numpy as np

from bhl_robust.eval import panels


def test_lidar_panel_puts_a_forward_wall_at_the_top():
    n = 36
    sectors = np.full(n, 12.0)
    # sector containing 0 deg (forward): edges at -180 + 10*i; forward is sector 18
    sectors[18] = 1.0
    img = panels.lidar_panel(sectors, 12.0, (240, 240), "lidar", window_m=6.0)
    a = np.asarray(img)
    h, w = a.shape[:2]
    cx = w // 2
    # the scan fill colour appears above the centre (forward) but not far below-left
    fill = np.array(panels.SCAN_FILL)
    above = np.abs(a[h // 2 - 30:h // 2 - 10, cx - 3:cx + 3].astype(int) - fill).sum(-1)
    assert (above < 30).any()


def test_depth_pair_and_compose_shapes():
    pair = np.random.default_rng(0).uniform(0.2, 6.0, size=(2, 8, 8))
    dp = panels.depth_pair_panel(pair, 6.0, (320, 170), "stereo pair (ray depth)", pooled=pair[:, ::2, ::2])
    assert dp.size == (320, 170)
    lp = panels.lidar_panel(np.full(36, 3.0), 12.0, (320, 200), "lidar", brake={"range_scale": 0.5})
    main = np.zeros((360, 640, 3), dtype=np.uint8)
    frame = panels.compose_frame(main, [dp, lp], "header", "footer", side_w=320)
    assert frame.dtype == np.uint8 and frame.shape[2] == 3
    assert frame.shape[1] == 960 and frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0
    stale = panels.depth_pair_panel(None, 6.0, (320, 170), "stereo", stale=True)
    assert stale.size == (320, 170)


def test_depth_colours_are_monotone_in_range():
    d = np.array([[0.0, 3.0, 6.0]])
    rgb = panels.depth_colours(d, 6.0).astype(int)
    assert rgb[0, 0].sum() > rgb[0, 1].sum() > rgb[0, 2].sum()
