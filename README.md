# Alireza-1
This is for mode classification

Run the combined model builder, solver, reports and modal audit from this folder:

```bat
cd /d "D:\CFS-Column\git_hub"

abaqus cae noGUI=abaqus_complete_model_m20.py -- ^
  --builtup-dir "C:\Users\810200014.HAMI.000\Documents\CUFSM-Single\cfs_abaqus\A4784_t3_qm1_0_0_0_R2p5_lipS_lipLen60_M80_L3600_gap10_nb19_end25-25_row15" ^
  --output-root "D:\CFS-Column\New folder (11)" ^
  --mesh-mm 5 ^
  --n-modes 250 ^
  --n-vectors 500 ^
  --max-iterations 1250 ^
  --cpus auto ^
  --gpus auto ^
  --buckle-output detailed ^
  --nodal-precision full ^
  --longitudinal-lines 2 ^
  --longitudinal-line-min-spacing-mm 5 ^
  --modal-audit
```


## Mechanical Stage-A modal classification

For new runs with `--modal-audit`, the primary classifier is now a whole-built-up-section, fcFSM-style **mechanical** decomposition. All four physical pieces enter one L/D/G definition, but no continuous tie, translational compatibility, rotational compatibility, artificial plate or stiffness is inserted across the physical gaps. The real discrete Abaqus bolts affect the observed eigenmode/eigenvalue; bolt spacing does not enter the reference-family definition.

The primary reported shares are `K0` elastic-energy `L/D/G/O`. `Assembly` is a separate relative-piece-motion diagnostic and is **not subtracted from the eigenmode and is not added to L/D/G/O**. Seam opening, transverse sliding and longitudinal slip are reported separately. The older geometry-based classifier remains in the same audit only as `geometric_screening_*` evidence.

New buckling runs store `U + UR`. A legacy ODB that contains only `U` remains readable for geometric screening but is explicitly marked `MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR`; missing rotations are never replaced by zeros.

Automatic audit flow:

```text
Abaqus eigenmode U+UR
  -> one-pass numeric archive
  -> canonical four-piece reference (no cross-gap constraints)
  -> multi-harmonic S-S decomposition
  -> K0/fcFSM L-D-G-O projection
  -> Assembly + seam diagnostics
  -> repeated-eigenvalue/eigenspace check
  -> CSV / JSON / HTML / piece-wise mode plots
```

The output `mode_sections.png` keeps the original gray section and independently colored pieces. Each panel now separates: mechanical family + K0 L/D/G/O, geometric screening, Assembly/seam diagnostics, and quality flags.

### Aggressive resources and progress

The default resource policy has no artificial CPU or RAM reserve. `--cpus auto` (also the default when `--cpus` is omitted) uses all detected logical CPUs; `--gpus auto` uses supported detected GPUs where the active Abaqus/numerical backend accepts them and otherwise records a CPU fallback. Abaqus Job memory is requested as 100 percent of host memory rather than the previous fixed 24000 MB ceiling. An explicit value such as `--cpus 8` remains a supported manual override.

Long operations print aggregate progress rather than worker spam: current stage, stage/overall percentage, exact `done/total` for countable work, elapsed time, rate, remaining stages and ETA. Abaqus solver progress that cannot be derived from an exact solver counter is visibly labeled `ESTIMATED` and is capped below 100% until successful completion is verified.

The automatic mechanical audit records reference/basis hashes, retained harmonics, resource layout and numerical backend in `modal_audit.json`. Synthetic checks and the CUFSM/fcFSM comparison harness are described in `README_modal_validation.md`. Stage A should not be treated as independently validated against CUFSM until that external benchmark has actually been executed for a compatible open-section case.


`--longitudinal-lines` controls the section boundaries retained during native
CAE geometry creation and meshing, before writing the INP:

- `0` (default): original simplification.
- `2` through `99`: up to that many internal lines per curved region, selected
  from the original geometry with greater density where turning is greater.
  Increasing the value retains previously selected lines.
- `100`: all original section lines, including collinear subdivisions.

`--longitudinal-line-min-spacing-mm` limits how close retained **nonessential**
longitudinal lines may be along the original section path. This includes both
new refinement lines and low-turn lines inherited from the virtual-topology
prepass. For example, `--longitudinal-line-min-spacing-mm 5` removes/avoids
soft boundaries that would create a strip narrower than 5 mm measured by
section arclength. The most curvature-sensitive line is retained when soft
lines compete inside the same spacing neighborhood. The default is `0`, which
preserves the previous behavior exactly.

Hard constraints are never removed: the two section-chain ends, exact bolt-row
partitions and genuine sharp corners (using the existing 10-degree merge
criterion) remain even if they are closer than the requested spacing. The
requested `--longitudinal-lines` count can therefore saturate below its budget
when no additional eligible source vertex remains. `--longitudinal-lines 100`
intentionally keeps its existing meaning and retains every source line,
bypassing the spacing filter. Value `1` is invalid.

The root builder includes the longitudinal-line changes from `code-change` and
retains the root pipeline's precision, reporting and audit features. Run the
root script. `--buckle-output detailed` remains a compatibility option: the
current buckling pipeline writes only nodal `U + UR` required by the mechanical classifier, not shell `S/E/SF/SE`.
Add `--build-only` to create CAE/INP without solving, or `--check-inputs` to
validate arguments and source CSVs without starting the model build.

Run the local regression tests with `python -m unittest discover -q`.

Reclassify an existing `modal_dsm_audit/modal_shapes.npz` with ordinary Python
(NumPy, SciPy and Matplotlib), without opening an ODB, building a model or
submitting a solver job:

```bat
python abaqus_reclassify_archive.py --run-dir "EXISTING_RUN" --output-dir "NEW_REPORT_DIRECTORY"
```

The output directory must not exist. Add `--modes 2,124,197` for a selected-mode
preview; incomplete eigenvalue clusters remain unresolved in that preview.
The full report includes geometry overlays, before/after percentages,
interactive shapes, eigenspace bounds, analytical controls and input hashes.
Eigenvalues and wavelengths remain those of the original calculation.

Source geometry now separates compact bends from complete curved/wavy walls.
Local displacement is measured against endpoint-driven wall motion that exactly
reproduces infinitesimal rigid rotation. The same interpolation drives the
coarse Distortional proxy. Both the normal pipeline and archive command use this
definition. These are nonorthogonal geometric screening components, **not a
validated mechanical L/D/G basis or energy partition**. See
[method and validation limits](docs/curved-wall-method.md) before interpreting
changes in the percentages.
