# ALE-BGK meshfree kinetic solver — lid-driven cavity

A meshfree Arbitrary Lagrangian–Eulerian solver for the BGK model of the
Boltzmann equation, applied to the rarefied lid-driven cavity with and without
an immersed rigid body.

Transport is semi-Lagrangian: the distribution function is reconstructed at
each particle's departure point by a moving least-squares fit over its
neighbours. Walls use diffuse reflection with the reflected Maxwellian scaled
so the discrete net mass flux vanishes. An immersed body carries its own
surface particles, which both impose its wall state and provide the quadrature
for the surface-stress integral that drives its motion.

## Cases

| # | Name | Description |
|---|------|-------------|
| 1 | `cavity2d` | 2-D lid-driven cavity |
| 2 | `cavity3d` | 3-D lid-driven cavity |
| 3 | `cavity2d-square` | 2-D cavity with a square body |
| 4 | `cavity2d-circle` | 2-D cavity with a circular body |
| 5 | `cavity3d-sphere` | 3-D cavity with a spherical body |
| 6 | `cavity3d-cube` | 3-D cavity with a cubic body |

In cases 3–6 the body is **free** by default: it translates and rotates under
the hydrodynamic force and torque. Pass `--fixed` to hold it in place, which
still integrates and reports its loads.

## Build and run

```bash
./Scripts/build.sh cuda      # or omit 'cuda' for the CPU backend
./build/bin/alebgk           # lists the cases
./build/bin/alebgk cavity3d-cube
./Scripts/run_all_cases.sh   # short runs of all six
```

Options: `--fixed`, `--Nx N`, `--Nv N`, `--dt T`, `--tfinal T`,
`--body-scale S`, `--tag NAME`.

Each case is fully specified in `TestCases/TestCases.cpp`. Naming a case is
enough to reproduce it — nothing is selected by environment variable.

## Layout

```
Core/                 particle and parameter types, output, checkpointing
Initialization/       lattice generation, initial state, neighbour lists,
                      body embedding
MLSTransport/         MLS basis and the semi-Lagrangian transport
MomentCalculations/   moments, BGK collision, conservation bookkeeping
BoundaryConditions/   wall state and diffuse reflection
RigidBody/            body geometry, dynamics, and cloud maintenance
Solver/               time loop; CUDA backend
TestCases/            the six case definitions and the entry point
Scripts/              build and run scripts, postprocessing
Figures/              generated figures
```

## Physical configuration

All six cases share a 1 µm domain, argon at 270 K, and diffuse walls.

| | plain cavity | with body |
|---|---|---|
| lid | uniform, 1 m/s | parabolic, peak 10 m/s |
| lid wall | y=L in 2-D, z=L in 3-D | y=L in both |
| gas density | 1.0 | 11.0 in 2-D, 4.4 in 3-D |

The 3-D body cases use a lighter gas than 2-D on purpose. At ρ=4.4 the mean
free path is λ/dx ≈ 0.96 and the Knudsen layer is resolved; the denser 2-D gas
leaves λ ≈ 0.25 dx unresolved, and the lid-edge wall fluxes then heat the gas
until the velocity grid truncates. Verified with a *fixed* body, so it is not
a body-coupling effect.

## Numerical choices worth knowing

Several parameters look arbitrary and are not. They are documented at the
point of use in `TestCases/TestCases.cpp`; in brief:

- **Time step.** `dt = 1.5e-11` for the body cases gives a foot-point CFL near
  1. A *smaller* step is unstable here: the MLS transport grows a cold dense
  spot from about 1000 steps at low CFL. This is a property of the base
  scheme, confirmed with a fixed body.
- **Velocity grid.** The 3-D body cases use `Nv = 10`, coarse by the usual
  standard (dv/σ = 1.22). Refining to `Nv = 20` made transport *less* stable,
  and changes the body's accumulated rotation by 0.14% — so the coarse grid
  costs nothing that matters.
- **Spatial grid.** A four-grid study (Nx = 31/40/50/60) puts the converged
  rotation at ≈65.3°, with Nx = 40 some 11% below it and Nx = 50 within 2.7%.
  The default is Nx = 40; raise it for production.
- **Reconstruction order.** Quadratic MLS, and it should stay that way. The
  scheme converges at second order (measured error ratio 0.546 against the
  0.640 a second-order scheme predicts). Dropping to linear is 1.84× faster
  per step but a net ~1.5× *slower* to any given accuracy.

## Checkpointing

Long runs write a resumable checkpoint:

```bash
ALEBGK_CHECKPOINT_EVERY=2000 ./build/bin/alebgk cavity3d-cube
ALEBGK_RESTART=output/cavity3d-cube_Nx40_Nv10/restart.ckpt ./build/bin/alebgk cavity3d-cube
```

The VTK snapshots hold moments only. For a kinetic solver the distribution
function *is* the state, so they cannot be resumed from — rebuilding it as a
Maxwellian would discard the non-equilibrium part, which is the physics being
solved. Restarts are bit-identical to an uninterrupted run.

## Postprocessing

`Scripts/postprocessing/` holds the analysis and figure scripts — vortex
detection, body trajectory and pose, convergence and limiter diagnostics.
Several recover a quantity independently of the solver's own log (body
rotation, for instance, by Kabsch alignment of the body's surface particles),
so a bug in one cannot hide behind the other.
