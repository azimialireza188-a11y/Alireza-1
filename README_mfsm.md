# Built-up mFSM implementation status

The numerical **supplied-operator mFSM energy decomposition** backend is implemented.
It is **not yet an automatically validated primary classifier for the production
four-piece S4R / BEAM_MPC / contact model**. The existing screening and independent
fcFSM reports remain available. Do not use the new numerical percentages as DSM
limit-state inputs before physical validation.

The kernel follows the generalized energy-ratio nullspace construction in
Khezri–Rasmussen (2019), Part 1, equations 125/134, and the separation of total
system stiffness from shell modal stiffness in the SSRC 2023 built-up-member
paper, equation 35. Shell strain components use auxiliary `nu_class=0`.
Connector stiffness enters `K_system`, never the shell strain components.
Source-specific admissible search spaces, warping constraints, closed-section
modified torsion and the L/D/G hierarchy must be independently supplied and
reviewed; a generic matrix nullspace alone does not establish that hierarchy.

## Run

Abaqus must provide U **and UR** for every mapped shell node. New builder runs
request both. Old U-only ODBs can still use screening; the mFSM adapter rejects
missing rotations instead of inventing them.

```bat
abaqus python abaqus_dsm_modal_audit.py --run-dir "D:\runs\actual" --classifier mfsm --mfsm-pack "D:\operators\actual.npz" --cpus auto --gpus auto
```

The combined CAE command also accepts `--modal-classifier mfsm --mfsm-pack PATH`.
The pack enables `--modal-audit` automatically. These options propagate through
`--resume-post RUN_DIR` without rebuilding or submitting the solver.

`auto` without an operator pack reports mFSM `UNAVAILABLE` with required evidence.
With a pack it generates a separate numerical decomposition whose status is
`NUMERICAL_DECOMPOSITION_AVAILABLE_PHYSICAL_VALIDATION_PENDING`. Neither case
silently replaces primary screening labels or marks the model scientifically
eligible. Explicit `mfsm` requires a pack; incompatible external/screening packs
are rejected.

## NPZ operator contract

Load uses `allow_pickle=False`; save JSON metadata as a scalar string.

| Array | Shape / meaning |
|---|---|
| `K_system` | n × n symmetric positive-definite auxiliary reduced system metric |
| `K_NAME` | n × n shell component operators named in metadata `components` |
| `mapping` | n × r; maps raw ODB U/UR to the documented reduced coordinates |
| `instances`, `labels`, `dofs` | length r; unique instance / node / DOF keys; DOFs 1..6 required per node |
| `coordinates` | r × 3; global undeformed node coordinates, repeated for each DOF |
| `raw_constraints` | optional c × r; checked on raw vectors **before** reduction |
| `H_FAMILY` | n × h documented admissible source search space |
| `C_FAMILY` | optional constraints on reduced coordinates |
| `metadata` | JSON scalar string |

Metadata requires `source_odb_sha256`, `model_signature`, `nu_class: 0`,
`coordinate_space_review`, `constraint_mapping_review`,
`contact_status: INACTIVE_VERIFIED|TANGENT_VERIFIED`,
`connection_status: ELASTIC_VERIFIED|RIGID_REDUCTION_VERIFIED`, `components`,
`source: {reference, equations, metric_definition}` (nonempty strings), and
`spaces: {FAMILY: {zero_components, equations}}`.
Family equations/search spaces are supplied mechanical definitions, **not**
visual labels. Names L/D/G and additional shear/transverse-extension spaces are
retained in the full denominator. Review flags establish declared provenance;
they do not themselves count as independently measured physical evidence.

`builtup_reference_section.py` preserves separate pieces/gaps, and
`shell_strain_operators.py` integrates supplied strain maps. These helpers do
**not** reconstruct an Abaqus S4R tangent. `builtup_connection_operator.py`
provides an ideal small-rotation rigid-link constraint kernel and elastic
connector assembly; it does **not** verify Abaqus BEAM_MPC semantics automatically.
Contact state/tangents and harmonic parity/mappings must be supplied explicitly.

## Results and validation

`mfsm_audit.json`, `mfsm_percentages.csv` and `mfsm_percentages.png` contain full
auxiliary-system energy shares, signed cross terms, closure/residual checks,
backend/cache/resource provenance and elapsed time. Shares can sum beyond 100%
when signed cross terms are present; closure includes those terms. Mixed and
unresolved states are QC; only LOCAL/DISTORTIONAL/GLOBAL are family labels.
Global subtype remains null without an independently verified mechanical
bending/torsion/warping decomposition.

Near-repeated modes use an orthonormal observed eigenspace, invariant trace shares
and minimum/maximum bounds. A cluster touching the last available eigenmode has
an open spectral boundary and cannot provide a stable family. Cache identity
includes the system, shell components, search spaces and source metadata;
writes are atomic and corrupted/mismatched cache data is rejected.

```sh
python -m unittest discover -q
python -m benchmarks.mfsm.run_performance --output timing.json
python -m benchmarks.mfsm.run_validation benchmarks/mfsm/pending_physical_validation.json --model-hash ACTUAL_HASH --output validation.json
```

The checked-in timing is synthetic, measured on this implementation host; it is
not an Abaqus benchmark or a guarantee of minimum runtime. The validation command
inventories missing evidence and always stays pending until an independent
artifact-based physical validation pipeline is implemented. No published example,
actual-model Abaqus run or GPU hardware comparison passed on this host.

Remaining implementation work also includes inverse/reconstruction checks for
the supplied reduction map, automatic integration of assembly/seam diagnostics,
live VRAM sizing and observed peak-memory telemetry. Current mapping validates
provenance, coordinates, dimensions and per-mode constraints; it does not measure
raw content discarded by a caller-supplied mapping matrix. See
[the source map](docs/mfsm-source-map.md) for implemented versus missing physics.
