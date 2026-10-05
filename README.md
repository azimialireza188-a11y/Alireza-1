# Alireza-1
This is for mode classification

Fast Step-4 candidate screening now reads actual ODB mode shapes with the same
`abaqus_step4_imperfections.py --run-dir "RUN" --suggest` command. See
[fast screening and execution](README_fast_modal_suggest.md) for the method,
CPU/GPU policy, cache, outputs and validation limits.

See [mFSM status and operator contract](README_mfsm.md) and
[full-resource execution](README_resource_execution.md). The new numerical backend
requires supplied documented operators and physical validation before primary
scientific/DSM activation.

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
current buckling pipeline writes `U` for classification, not shell `S/E/SF/SE`.
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
