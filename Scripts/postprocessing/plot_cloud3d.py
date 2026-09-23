#!/usr/bin/env python3
"""ParaView-style particle-cloud rendering: the raw meshfree cloud drawn
as shaded sphere glyphs coloured by density (ParaView's Point Gaussian
look -- Cool-to-Warm colormap, classic gray-blue background, orientation
axes, scalar bar). Works for any run's VTK series, 2D or 3D.

Usage:
  python3 postprocessing/plot_cloud3d.py <outprefix> [options] <a.vtk> ...
Options:
  --scalar S      point scalar to colour by (default rho); the special
                  value "speed" colours by |velocity| computed from the
                  velocity vector field
  --psize P       glyph size in screen pixels (default 7)
  --camera C      pyvista camera preset: xy | xz | yz | iso (default xy,
                  the ParaView default front view)
  --azel A E      extra azimuth/elevation in degrees after the preset
                  (default 15 8 -- a slight ParaView-like tilt that avoids
                  the lattice moire of a dead-on view; use 0 0 to disable)
  --clim LO HI    fixed colour range (default: global range of the inputs)
  --style S       points (default) | arrows | stream:
                    arrows -- 3D quiver: arrow glyphs oriented/scaled by the
                              local velocity, coloured by speed
                    stream -- 3D streamlines traced through the binned
                              velocity field, drawn as tubes coloured by
                              speed, seeded throughout the volume
  --every N       (arrows) use every Nth particle as a glyph seed (default 1)
  --seeds N       (stream) number of streamline seed points (default 150)
  --grid          render all snapshots as panels of ONE image (2 columns,
                  3 for more than 4 inputs; shared colour range, scalar
                  bar on the last panel) instead of one file per snapshot

Requires pyvista; needs a display (DISPLAY=:0/:1) or Xvfb.
Writes <outprefix>_t<time>.png per snapshot.
"""
import sys

import numpy as np
import pyvista as pv

PV_BACKGROUND = (0.32, 0.34, 0.43)     # ParaView's classic background


def also_save_eps(png_path):
    """Wrap a rendered PNG into an EPS twin (raster-in-EPS) so the
    figure drops into dvips/epsfig LaTeX builds unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(png_path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 200.0, h / 200.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    eps = png_path.rsplit(".", 1)[0] + ".eps"
    fig.savefig(eps, dpi=200)
    plt.close(fig)
    print(f"saved {eps}")


def read_vtk(path, scalar):
    with open(path) as f:
        lines = f.read().split("\n")
    t = None
    if len(lines) > 1 and "t=" in lines[1]:
        t = float(lines[1].split("t=")[-1].strip())
    pts = val = vel = None
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if tok and tok[0] == "POINTS":
            n = int(tok[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            pts = np.array(vals).reshape(-1, 3)
            continue
        if tok and tok[0] == "SCALARS" and tok[1] == scalar:
            n = pts.shape[0]
            vals = []
            i += 2
            while len(vals) < n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            val = np.array(vals)
            continue
        if tok and tok[0] == "VECTORS":
            n = pts.shape[0]
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            vel = np.array(vals).reshape(-1, 3)
            if scalar == "speed":
                val = np.linalg.norm(vel, axis=1)
            continue
        i += 1
    return t, pts, val, vel


def add_flow(p, pts, val, vel, scalar, style, psize, clim,
             every=1, seeds=150, bar=True):
    """Add the chosen flow representation to plotter p."""
    if style == "arrows":
        sub = slice(None, None, max(1, every))
        pd = pv.PolyData(pts[sub])
        pd["velocity"] = vel[sub]
        pd["speed"] = np.linalg.norm(vel[sub], axis=1)
        L = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        # Uniform arrow length (~1.6 lattice cells), colour carries the
        # magnitude -- scaling by speed makes the slow interior invisible
        # next to the lid (10x speed ratio in the cavity).
        n3 = round(len(pts) ** (1.0 / 3.0))
        factor = 1.6 * (L / max(n3 - 1, 1))
        glyphs = pd.glyph(orient="velocity", scale=False, factor=factor)
        p.add_mesh(glyphs, show_scalar_bar=bar, scalars="speed", cmap="coolwarm", clim=clim,
                   scalar_bar_args=dict(title="speed", color="white",
                                        vertical=True, position_x=0.88,
                                        position_y=0.25, height=0.5,
                                        width=0.05))
        return
    if style == "stream":
        from scipy.ndimage import zoom
        n = pts.shape[0]
        G = max(8, int(round(n ** (1.0 / 3.0))))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        idx = np.clip(((pts - lo) / span * G).astype(int), 0, G - 1)
        sv = np.zeros((G, G, G, 3))
        c = np.zeros((G, G, G))
        for k in range(n):
            sv[idx[k, 0], idx[k, 1], idx[k, 2]] += vel[k]
            c[idx[k, 0], idx[k, 1], idx[k, 2]] += 1
        c = np.maximum(c, 1)
        gv = sv / c[..., None]
        up = 2
        gf = np.stack([zoom(gv[..., d], up, order=2) for d in range(3)],
                      axis=-1)
        img = pv.ImageData(dimensions=gf.shape[:3],
                           spacing=tuple((hi - lo) /
                                         (np.array(gf.shape[:3]) - 1)),
                           origin=tuple(lo))
        img["velocity"] = gf.reshape(-1, 3, order="F")
        ctr = tuple(0.5 * (lo + hi))
        L = float(np.max(hi - lo))
        sl = img.streamlines(vectors="velocity", n_points=seeds,
                             source_center=ctr, source_radius=0.42 * L,
                             integration_direction="both",
                             max_time=50.0 * L, terminal_speed=1e-12)
        if sl.n_points:
            sl["speed"] = np.linalg.norm(sl["velocity"], axis=1)
            tubes = sl.tube(radius=L / 350.0)
            p.add_mesh(tubes, show_scalar_bar=bar, scalars="speed", cmap="coolwarm", clim=clim,
                       smooth_shading=True,
                       scalar_bar_args=dict(title="speed", color="white",
                                            vertical=True, position_x=0.88,
                                            position_y=0.25, height=0.5,
                                            width=0.05))
        p.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1],
                                  lo[2], hi[2])),
                   style="wireframe", color="white", line_width=1,
                   opacity=0.4)
        return
    # default: point cloud
    cloud = pv.PolyData(pts)
    cloud[scalar] = val
    p.add_mesh(cloud, show_scalar_bar=bar, scalars=scalar, cmap="coolwarm", clim=clim,
               render_points_as_spheres=True, point_size=psize,
               scalar_bar_args=dict(title=scalar, color="white",
                                    vertical=True, position_x=0.88,
                                    position_y=0.25, height=0.5,
                                    width=0.05))


def main():
    args = sys.argv[1:]
    prefix = args.pop(0)
    scalar, psize, camera, clim = "rho", 7.0, "xy", None
    az, el = 15.0, 8.0
    grid = False
    style, every, seeds = "points", 1, 150
    while args and args[0].startswith("--"):
        k = args.pop(0)
        if k == "--scalar":
            scalar = args.pop(0)
        elif k == "--psize":
            psize = float(args.pop(0))
        elif k == "--camera":
            camera = args.pop(0)
        elif k == "--azel":
            az = float(args.pop(0)); el = float(args.pop(0))
        elif k == "--style":
            style = args.pop(0)
        elif k == "--every":
            every = int(args.pop(0))
        elif k == "--seeds":
            seeds = int(args.pop(0))
        elif k == "--grid":
            grid = True
        elif k == "--clim":
            clim = (float(args.pop(0)), float(args.pop(0)))

    snaps = []
    for f in args:
        t, pts, val, vel = read_vtk(f, scalar)
        if style != "points":
            val = np.linalg.norm(vel, axis=1)      # colour by speed
        if val is None:
            print(f"  {f}: scalar '{scalar}' not found -- skipped")
            continue
        snaps.append((t, pts, val, vel))
        print(f"  {f}: t={t:g}  range [{val.min():.3f},{val.max():.3f}]")
    if clim is None:
        clim = (min(s[2].min() for s in snaps),
                max(s[2].max() for s in snaps))

    if grid:
        n = len(snaps)
        ncols = 2 if n <= 4 else 3
        nrows = (n + ncols - 1) // ncols
        p = pv.Plotter(off_screen=True, shape=(nrows, ncols),
                       window_size=(950 * ncols, 800 * nrows), border=False)
        for i, (t, pts, val, vel) in enumerate(snaps):
            p.subplot(i // ncols, i % ncols)
            p.set_background(PV_BACKGROUND)
            add_flow(p, pts, val, vel, scalar, style, psize, clim,
                     every, seeds, bar=(i == len(snaps) - 1))
            p.add_text(f"t = {t:g}", font_size=12, color="white")
            p.camera_position = camera
            p.camera.azimuth = az
            p.camera.elevation = el
            p.camera.zoom(1.25)
        out = f"{prefix}_grid.png"
        p.screenshot(out)
        p.close()
        print(f"saved {out}")
        also_save_eps(out)
        return

    for t, pts, val, vel in snaps:
        p = pv.Plotter(off_screen=True, window_size=(1500, 1100))
        p.set_background(PV_BACKGROUND)
        add_flow(p, pts, val, vel, scalar, style, psize, clim, every, seeds)
        p.add_text(f"t = {t:g}", font_size=13, color="white")
        p.add_axes(color="white")
        p.camera_position = camera
        p.camera.azimuth = az
        p.camera.elevation = el
        p.camera.zoom(1.3)
        out = f"{prefix}_t{t:g}.png"
        p.screenshot(out)
        p.close()
        print(f"saved {out}")
        also_save_eps(out)


if __name__ == "__main__":
    main()
