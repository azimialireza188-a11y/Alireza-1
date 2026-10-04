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
