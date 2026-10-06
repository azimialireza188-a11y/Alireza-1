# Built-up mFSM implementation status

The numerical **operator-based mFSM energy decomposition** backend is implemented,
including an automatic 2019 modal hierarchy on reviewed component operators.
It is **not yet an automatically validated primary classifier for the production
four-piece S4R / BEAM_MPC / contact model**. The existing screening and independent
fcFSM reports remain available. Do not use the new numerical percentages as DSM
limit-state inputs before physical validation.

The kernel follows the generalized energy-ratio nullspace construction in
Khezri–Rasmussen (2019), Part 1, equations 125/134, and the separation of total
system stiffness from shell modal stiffness in the SSRC 2023 built-up-member
paper, equation 35. Shell strain components use auxiliary `nu_class=0`.
Connector stiffness enters `K_system`, never the shell strain components.
The new `KHEZRI_2019_HIERARCHY` path constructs GA, GB, GT (or modified GT),
D, L, TE and aggregate S using the complete documented coordinate space,
warping orthogonality and transverse equilibrium. It needs reviewed six-component
operators and canonical equilibrium/topology data; it does not derive physical
operators from an ODB. The earlier supplied-search-space path remains supported.

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
| `reconstruction` | r × n; compatible right inverse, `mapping @ reconstruction = I` |
| `raw_metric_diagonal` | r positive dimensional norm weights; document translation/rotation unit scaling |
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
`spaces: {FAMILY: {zero_components, equations}}` for the supplied-search-space path.
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

## Automatic source hierarchy

Set `basis_construction: "KHEZRI_2019_HIERARCHY"` and omit `spaces`, `H_FAMILY`,
`C_FAMILY` and supplied `global_subspaces`. All six components are required:
`eps_x`, `eps_y`, `gamma_xy`, `kappa_x`, `kappa_y`, `kappa_xy`. `K_system`
must already be reduced to positive energetic coordinates; no hidden pins or
extra gap ties are introduced. Provide metadata such as:

```json
"hierarchy": {
  "coordinate_definition": "Reviewed complete canonical/reduced DOF ordering and harmonics",
  "equilibrium_definition": "Reviewed Appendix A1 node/DOF selection and reduction",
  "equilibrium_construction": "SUPPLIED_Z_TE",
  "closed_cells": false,
  "open_shear_selection": "WARPING",
  "warping_projection_review": "Reviewed canonical V-only projection in these coordinates"
}
```

`Z_te` is an e × n array of transverse equilibrium covectors. Appendix A1 uses
rows of **canonical `K_kappa_x`**, corresponding to theta at main nodes and
U/W/theta at internal nodes; it does not select arbitrary reduced coordinates.
The alternative `equilibrium_construction: "APPENDIX_A1_ROWS"` builds it from
`K_kappa_x_canonical`, `canonical_reduction` (Q) and integer `unprescribed_rows`.
It checks `Q.T @ K_kappa_x_canonical @ Q == K_kappa_x`, then selects canonical
rows before multiplying Q. Do not also supply `Z_te` for that alternative.
`fsm_unprescribed_rows(node_count, main_nodes, harmonics)` generates row indices
only for the explicit harmonic-major/node-major U,V,W,theta FSM ordering.
Main nodes must come from reviewed physical topology; the helper does not infer
them from mode images or faceted curvature.

For a general coordinate change, supply n × n `coordinate_metric`, the pullback
of canonical Euclidean orthogonality (e.g. `Q.T @ Q`). The Appendix path derives
this metric from Q. Identity in the supplied-Z path declares canonical
orthonormal coordinates; it is not valid for every reduction/rescaling.

If the constituent strip topology actually contains closed cells, set
`closed_cells: true`, document `closed_loop_review` and supply `K_gamma_open`,
the shear operator assembled **only from strips outside the closed loops**
(Part 2 equations 2–4). Its shape/PSD and the remaining closed-strip shear PSD
are checked. The full shear and all supported fastener effects remain in the
system metric. A box-like built-up outline or discrete bolt seams alone do not
establish the continuous closed-loop strip topology needed by this construction.

Two explicit numerical adaptations remain subject to physical validation:
Part 2 Eq12 is diagonalized within its candidate space to separate positive
curvature from pure warping shear invariantly; Part 1 Eq200 uses the aggregate
transverse shear space (null eps_x/eps_y, independent of L) instead of interpreting
the printed SDw/SDt ambiguity in Eq188–189. See the source map. GA is retained
as internal extension, and **G contains GB+GT**, excluding GA.

The source hierarchy also builds mechanical FLEXURAL/TORSIONAL subtype bases.
The existing residual/cross-term/repeated-cluster QC still applies. No scientific
activation is granted by either hierarchy construction or review flags.

For open sections, shear selection is explicit, as required by the open row of
Part 2 Table 1. `open_shear_selection: "WARPING"` requires an n × n
`warping_projection` array that retains canonical longitudinal V DOFs and zeros
all other DOFs. It must be idempotent, self-adjoint in the canonical coordinate
metric, and consistent with the six strain maps. This path builds the aggregate
Sw (including its associated/secondary/complementary warping shear) and SCt from
the literal Part 1 Eq188–189. For a changed basis T, transform P as `T^-1 P T`.
An exact-MPC reduction must preserve this projection; otherwise use independently
reviewed compatible source spaces instead of inventing a reduced V mask.

The alternative `open_shear_selection: "REVIEWED_AGGREGATE"` requires `S_open`
(n × s) and nonempty metadata `open_shear_equations`. This supports a different
reviewed Table 1 selection, including transverse choices; G/D/L remain automatic,
but the auxiliary open S basis is supplied explicitly in this alternative.
The closed-section orthogonal-complement rule is never used for open sections.
For closed G, modified GT is used; D keeps the classical Part 1 GT exclusion,
as specified by the unchanged RD entry in Part 2 Table 1.

## Results and validation

`mfsm_audit.json`, `mfsm_percentages.csv` and `mfsm_percentages.png` contain full
auxiliary-system energy shares, signed cross terms, closure/residual checks,
backend/cache/resource provenance and elapsed time. Shares can sum beyond 100%
when signed cross terms are present; closure includes those terms. Mixed and
unresolved states are QC; only LOCAL/DISTORTIONAL/GLOBAL are family labels.
For the earlier supplied-search-space path, global subtype remains null without
supplied reviewed mechanical flexural/torsional spaces. To provide them, include metadata `global_definition_review: true`,
`global_source: {reference, equations, metric_definition}`, and
`global_subspaces: ["FLEXURAL", "TORSIONAL"]`, plus `G_FLEXURAL` / `G_TORSIONAL`
arrays in reduced coordinates. Their metric must match the main projector and
their span must lie inside G. Per-mode and eigenspace subtype QC retain null when
content is unresolved or a repeated eigenspace has no unique subtype. These
inputs do not constitute independent scientific validation.

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

## Reconstruction, diagnostics and report integrity

A supplied reduction map now needs `reconstruction` and `raw_metric_diagonal` to
obtain resolved numerical labels. Both must be supplied together. The inverse
identity is checked; each raw mode is reconstructed and compared in the supplied
dimensional U/UR norm. `mapping_tolerance` defaults to 1e-8. Missing or failed
reconstruction yields UNRESOLVED with no dominant family/subtype; it also clears
stable cluster labels. Explicitly record how rotation weights convert radians
to a norm compatible with translation units in pack provenance.

Assembly diagnostics use the raw U of each mapped node, with instance names as
piece identifiers, and never subtract relative piece motion before mechanical
classification. Optional `diagnostic_seams` entries contain `a` / `b` keys as
`[instance, label]` and `axes` as a 3x3 orthonormal matrix whose rows are opening,
transverse-slip and longitudinal-slip directions. Components are signed raw
amplitudes after removal of common rigid motion; arbitrary eigenvector scaling
must be considered before comparing runs. `mfsm_mapping_assembly.csv` contains
mapping QC and diagnostics with their separate nodal norm denominator.

Every artifact is rendered in a staging directory before any target replacement.
Files are atomically replaced individually, and JSON is committed last with
SHA256 hashes of its artifacts. `mfsm_audit.load_report(directory)` verifies
those hashes and rejects interrupted/mixed generations. This detects an
interruption during multiple file replacements; it is not a filesystem-wide
transaction. Existing runs still use new output directories.

Live VRAM-based chunk sizing, smaller-batch allocation retries, lifetime process
RSS/peak and GPU allocator snapshots are now implemented. Measured CPU topology
tuning and automatic S4R/MPC/contact extraction remain unfinished. The 2019
hierarchy is implemented on supplied reviewed operators, with the documented
adaptations and physical benchmarks pending. See [the source map](docs/mfsm-source-map.md).

Failed or missing reconstruction is handled before numerical projection, including
fully discarded modes and rank-collapsed repeated clusters. The audit returns
UNRESOLVED rows/clusters and continues reporting other valid modes. For a repeated
cluster with indeterminate global subtype, accepted per-mode `global_subtype` is
cleared; `observed_global_subtype` retains basis-dependent content as diagnostic
and is explicitly flagged. The accepted subtype CSV column follows that gate.

The new source-hierarchy subtype identifier binds names, metric and main basis
identity. The earlier supplied-subtype path still has the deferred provenance
polish noted in the implementation ledger; neither path activates acceptance.

## Actual INP inspection and portable ODB transfer

`abaqus_mfsm_input.py` reads expanded global-coordinate S4R INPs, applies
instance translation before rotation, resolves singleton BEAM node sets and
assembles a sparse **initial** constraint matrix. Numeric homogeneous initial
BCs or explicit BUCKLE `LOAD CASE=2, OP=NEW` BCs are supported. Unsupported
MPCs, multiple-node pairing, transformed nodal axes, includes, additional
constraint forms and general preload steps fail explicitly. It creates no
contact tangent, shell stiffness or dense nullspace, and never marks active
base-state equivalence as verified.

```bat
python abaqus_mfsm_input.py --inp "BU_BOLT_L3600_M20(1).inp" --output-dir "input_audit_new"
```

Requires NumPy/SciPy; writes `input_audit.json`, sparse `initial_constraints.npz`
and `initial_dof_map.npz`. Uploaded actual input inspection is recorded in
[docs/mfsm-actual-input-audit.json](docs/mfsm-actual-input-audit.json): 40,700 nodes,
39,744 S4R elements, 76 BEAM MPCs, 1,777 constraint rows and 2,537 nonzeros.
Contact is defined, but its active base-state tangent remains unknown. The ODB
attachment failed transfer; no actual eigenmode was read on this host.

To avoid transferring a large binary ODB, run the following **with Abaqus Python**
on the machine containing Abaqus and the ODB. Supply actual absolute paths as
needed; `modal_export_new` must be a new directory.

```bat
abaqus python abaqus_mfsm_export.py --odb "BU_BOLT_L3600_M20.odb" --inp "BU_BOLT_L3600_M20(1).inp" --output-dir "modal_export_new"
```

Send `modal_export.json`, `raw_dof_map.npz` and the `modes_*.npz` files from that
directory. NumPy can read these without Abaqus/odbAccess. Default 8-mode shards
reduce individual transfer size; `--modes-per-shard` changes transfer granularity,
not memory reservation. The exporter now prefers Abaqus `FieldBulkData` blocks
for U/UR extraction and builds the node-label mapping once for the entire run;
the legacy per-`FieldValue` path is only a compatibility fallback. Shard
compression is pipelined across all visible logical CPUs while one process owns
the ODB handle. No lower RAM/VRAM percentage ceiling is introduced and allocation
backoff occurs only after an actual `MemoryError`. Live progress reports the
bulk/fallback counts, mode rate, ETA and pending compression work. Use
`--compression store` when minimum local export wall time is more important than
transfer size; the default `zlib` retains smaller upload artifacts. GPU execution
is intentionally not used for native ODB I/O/ZIP compression because this stage
has no supported GPU numerical kernel; GPU selection remains in the downstream
mFSM numerical stages. Output is raw data, **not** an mFSM operator pack or
validated classification.

The exporter checks exact instance/node/element connectivity and source-coordinate
agreement, reads global U and UR without zero fill, preserves negative eigenvalues
and original precision metadata, and hashes input/ODB/output artifacts. FP64
storage cannot recover precision lost in the original analysis. PRESELECT in an
INP does not by itself prove missing UR: actual ODB fields must be inspected.
Existing output directories are rejected; artifacts are staged and JSON committed
last. Failed exports have no final manifest and must not be treated as complete.
Actual Abaqus execution remains untested here; unit tests exercise synthetic ODB
objects and failure paths. Physical S4R/contact operators and validation remain
unfinished even after a successful raw-data export.

The supported INP keyword subset is explicit: nonrectangular NODE coordinates,
external data and unimplemented geometry-generation/transformation keywords are
rejected. CAE's NSET INTERNAL flag is accepted as set metadata.
Close the analysis/ODB writer before export. The CLI rejects an existing `.lck`
file without deleting it, snapshots source hashes and file metadata, and checks
both INP and ODB again before publication. Changed sources abort without a final
manifest. This detects observed changes; it does not acquire an exclusive source
lock. Topology agreement does not establish material/load/history equivalence.
The in-memory test API has `source_stability_verified=false` unless a successful
source guard is supplied; the CLI supplies this guard.

### Recovered production modal data (2026-10-04)

The subsequent recovered Parquet archive was received and inspected successfully;
the raw ODB transfer failure no longer blocks access to its exported mode vectors.
[Actual portable audit](docs/mfsm-actual-portable-audit.json) records all36 verified
artifact hashes,250 complete FP64 U/UR modes,40,700 matching nodes and39,744
matching S4R elements. Every shard was read and all values checked for finiteness.
The source INP hash matches the previously inspected input. Raw node-major DOFs
were reordered by instance/label/DOF before applying the sparse initial C.
The worst group-relative BEAM translation residual was3.220843507406831e-15;
BEAM rotation and mode boundary-condition residuals were zero. These checks
validate initial algebraic compatibility, not active-contact tangent equivalence.

Exported eigenvalues range from268.78 to391.20 and retain the precision present
in the exported mode descriptions. They are load multipliers here; no stress
units, L/D/G classifications or DSM eligibility are inferred. The raw ODB hash
and source-stability assertion remain exporter claims, since only the portable
archive was read. Actual S4R/component stiffness, active contact and physical
harmonic mapping remain necessary before a production mFSM decomposition.

### Automatic source mechanical preparation

`prismatic_mfsm_operators.py` extracts a strictly extruded GLOBAL-Z cross section
from expanded S4R connectivity without merging pieces across gaps. It constructs
sparse auxiliary nu=0 operators from Part1 (2019) Eqs1-19: linear membrane, cubic
Hermite bending, engineering twist, exact longitudinal sine/cosine orthogonality
and four-point transverse Gauss integration. These are faceted CPT operators;
Abaqus S4R stabilization, drill stiffness and transverse shear are not recreated.
Canonical node DOFs are UX,UZ,UY,URZ. A source-map sign test checks URZ versus
Hermite slope; full six-DOF reconstruction uses CPT slopes and facet-average
membrane-curl drill rotations. That additional reconstruction is an explicitly
unvalidated S4R mapping approximation.

`portable_mfsm_preparation.py` reads and verifies the recovered Parquet format,
checks mesh/hash/DOF/mode coverage, preserves signed eigenvalues, fits all modes
with shared sine/cosine factorizations and checks inverse reconstruction and
auxiliary component energy across harmonic enrichments. The default grows from
one term to the station-resolved sine dimension; it has no arbitrary term cap.
A separate constant-along-length warping field V0 extends the positive-harmonic
source series. Its only strain energy is gamma_xy and uniform V0 remains a
rigid-null field. This extension, facet averaging and the equal-section-node,
trapezoid-station reconstruction norm (rotation length = thickness) are declared
adaptations, not claims of reproduced paper benchmarks.

```bat
python -m pip install numpy scipy pyarrow threadpoolctl
python portable_mfsm_preparation.py --portable-dir "portable_modal_export_recovered" --inp "BU_BOLT_L3600_M20(1).inp" --output-dir "mechanical_preparation_new"
```

Output is `mechanical_preparation.json` in a new directory. Input files are read
only; source hashes are checked again before JSON-last publication. Missing
rotations, nonfinite fields, inconsistent topology and changed sources fail
explicitly. The sparse constraint projector applies the exact Euclidean action
I-C.T(CC.T)^-1C on supported independent rows without allocating a dense raw Q.
It is an initial kinematic diagnostic, not a contact tangent or energy metric.

Assembly runs on all visible CPU processes, weighted inverses are factored once
per parity, and reusable sparse matrices serve every mode. The numerical stage
measures all-core BLAS versus all-core worker throughput and chooses the faster
observed arrangement. Optional CuPy tests representative FP64 energies and
reconstruction against CPU including transfers, and schedules all measured-faster
GPUs; raw fields remain resident across harmonic levels. Allocation failure
retries smaller mode subsets; GPU allocation failure has a recorded CPU fallback.
Representative probe reads also reduce their Parquet column selection after actual
allocation failure. Optional CPU topology tuning measures the live memory-fitting
worker counts and selected BLAS budget; if concurrent tuning allocations fail,
it retries fewer workers and can retain the feasible all-core serial run.
No RAM/VRAM reserve percentage or standing savings ceiling is applied. Reported
RSS/peak covers the parent process; process children are not included. Actual
GPU/Windows execution remains unverified on this host.

[Production preparation measurement](docs/mfsm-actual-mechanical-preparation.json):
250 modes,9 visible CPUs,no GPU,1..183 harmonics,about12 seconds including assembly
and topology probes,2.27GiB parent peak RSS. Sample throughput improved from
0.881 to0.154 seconds (about5.7x); this is a measured preparation-stage comparison,
not the total mFSM classifier runtime or a guarantee on another machine. The
post-review run took12.016 seconds and its sample comparison was about5.4x.
These measurements preceded external scratch cleanup; the recovered code passes
fresh regression tests, but the250-mode archive could not be rerun after cleanup.
All
six-DOF reconstruction residuals were below0.44%, but247 modes changed auxiliary
energy by more than0.5% between128 and183 terms. Those illustrative summaries
are not AISI thresholds and do not establish strain-energy convergence.

**Production L/D/G transition is still incomplete.** This command deliberately
returns null families and UNAVAILABLE: it has not generated the actual admissible
source hierarchy, verified active contact, reproduced physical S4R/curved
benchmarks, or validated energy convergence. Source component energies must never
be renamed L/D/G percentages or used as DSM minima.

### Coupled initial harmonic constraint diagnostics

The preparation CLI now composes the actual initial BEAM/BC matrix with the
declared six-DOF harmonic reconstruction: C_h = C_raw R. Only raw DOFs used by
constraints are evaluated; all harmonic columns and the separately labeled V0
are retained together. End-condition zero rows are identified without assuming
independence. The largest map is constructed once and smaller harmonic levels
reuse its column subsets. CPU/GPU parity probes include the new residuals.
Each harmonic level reports its worst row-relative constraint cancellation
residual and the change in each auxiliary strain-component energy. No mode or
energy is silently corrected to improve either diagnostic.

[Synthetic scale measurement](docs/mfsm-synthetic-constraint-mapping.json):
1,777 constraints,244,200 raw DOFs and161,260 harmonic/V0 coordinates; the
mapping took0.975 seconds and its CSR storage was4,657,640 bytes. This synthetic
geometry only matches counts; it is not the actual production member or a
classification benchmark. Full-coordinate rank/nullspace, physical S4R/contact
equivalence and final family decomposition remain unverified. The earlier
12-second production measurement predates this added diagnostic; updated
production elapsed time has not been measured because scratch maintenance
removed the uploaded INP/archive from the execution workspace.

## Portable Parquet analysis with a documented operator pack

The portable export can now use the same numerical classifier and report engine
as the ODB adapter, without installing Abaqus:

```bat
python portable_mfsm_analysis.py --portable-dir "D:\exports\actual" --inp "D:\models\actual.inp" --mfsm-pack "D:\operators\actual.npz" --output-dir "D:\reports\mfsm-new"
```

Install NumPy, SciPy, pyarrow, matplotlib and threadpoolctl in that Python
environment. CuPy is optional for a compatible CUDA runtime. The command uses
all visible CPUs and considers all visible GPUs under the existing measured
FP64 backend policy, with zero RAM/VRAM reserves. Allocation failures retry a
smaller mode batch; this is capacity recovery, not a standing utilization cap.
The output directory must be new. `--cache-dir PATH` optionally selects the
existing content-addressed basis cache.

The NPZ schema is the documented supplied-operator schema above, additionally
requiring `metadata.source_inp_sha256` to match the exact input bytes. Its raw
map must include every portable node and all six DOFs; its order may differ.
The export's source ODB hash, global coordinates, complete topology, artifact
hashes, U/UR fields and mode table are checked before analysis. Independently
compiled initial BEAM/BC constraints are applied even if the pack omits
`raw_constraints`. Unsupported initial constraints fail explicitly.

Outputs are `mfsm_audit.json`, `mfsm_percentages.csv`,
`mfsm_mapping_assembly.csv` and `mfsm_percentages.png`, with checksums in the
JSON commit record. Eigenvalues retain their signed exported values. Close
clusters use the existing invariant eigenspace analysis; the last observed
spectral boundary remains open. Source files are checked again before the
report is published. Read reports with `mfsm_audit.load_report` to verify hashes.

This completes the portable **supplied-operator execution route**. It does not
construct the production S4R/contact tangent or the complete scalable
compatible family hierarchy. Supplying review flags is not independent
physical evidence. Results retain `scientifically_eligible=false` and cannot
be treated as validated AISI/DSM inputs. The actual-model physical classifier
and benchmark activation remain unfinished as recorded in the status document.
