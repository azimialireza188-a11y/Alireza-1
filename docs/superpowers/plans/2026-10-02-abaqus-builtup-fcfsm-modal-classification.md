# Abaqus Built-Up fcFSM Modal Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mechanically defined, whole-built-up-section L/D/G modal classifier for Abaqus eigenmodes, with separate Assembly/seam diagnostics and aggressive all-resource execution.

**Architecture:** Abaqus remains the eigensolver and stores `U + UR`. A canonical four-piece reference section, independent of bolt spacing, generates fcFSM-style `K0/J_GD/J_D` family subspaces. ODB data are extracted once, then harmonic decomposition, mechanical projection, Assembly/seam diagnostics, eigenspace checks, and plotting run in an aggressively parallel post-processing pipeline with CPU/GPU capability detection.

**Tech Stack:** Abaqus 2024 Python, CPython 3-compatible Python, NumPy, SciPy sparse linear algebra, Matplotlib, optional CuPy when present and numerically equivalent, CUFSM 5.70 fcFSM MATLAB source as the algorithmic reference.

**Spec:** `docs/superpowers/specs/2026-10-02-abaqus-builtup-fcfsm-modal-classification-design.md`

## Global Constraints

- Treat the full four-piece built-up section as one L/D/G system.
- Do not add continuous ties, translational compatibility, rotational compatibility, or artificial strips across physical gaps between pieces.
- Actual discrete BEAM MPC bolts remain only in the Abaqus physical model; bolt spacing/count must not enter the family-definition/K0 hash.
- Assembly is an independent diagnostic and must not be subtracted before L/D/G classification.
- Primary metric: `K0` elastic strain energy; secondary metric: dimensionally consistent kinematic/vector check.
- New mechanical runs require both `U` and `UR`; legacy U-only ODBs must fail mechanical classification explicitly, while geometric screening remains available.
- Automatic Buckle output stays lean: do not re-enable `S/E/SF/SE` or automatic shell-energy processing.
- Preserve existing CLI forms, CAE/INP generation, `--build-only`, `--resume-post`, Step 4 imperfections, and Step 5 GMNIA compatibility.
- Default resource behavior is aggressive: use all detected logical CPUs, request maximum supported Abaqus memory, use supported GPUs when numerically equivalent, and maintain no artificial CPU/RAM reserve.
- Explicit positive user CPU/GPU caps remain manual overrides for backward compatibility.
- Stage B restricted-family eigenproblems are not part of this plan.
- Every long-running path must emit aggregate progress: current stage, overall percent, exact done/total where countable, remaining stages, elapsed time, and ETA. Solver-only heuristic percentages are labeled ESTIMATED and never reach 100% before verified completion.

## Review Focus

- **Disconnected four-piece K0 / rigid nullspace:** reference construction must not regularize singularity by cross-gap ties; Task 4 adds a rigid-nullspace regression.
- **Independent piece motion that visually resembles D:** L/D/G classification must use the full original mode while Assembly is only diagnostic; Task 5 adds this regression.
- **Repeated eigenvalues under arbitrary eigensolver rotation:** family bounds must be eigenspace-invariant; Task 7 adds rotated-cluster tests.
- **U-only legacy ODB:** mechanical classification must report `MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR`, not insert zero rotations; Task 2 adds this regression.
- **Aggressive parallel/GPU execution changing numerical answers:** parallel CPU and optional GPU results must match deterministic CPU serial reference within explicit tolerance; Tasks 1 and 6 add equivalence tests.

---

## File Structure

**Create**
- `abaqus_progress.py` — stage-weighted progress, exact counters, ETA/rate, solver-estimate history.
- `test_progress.py`
- `abaqus_resource_policy.py` — resource detection, CPU/BLAS layout, GPU capability, aggressive resource provenance.
- `abaqus_modal_archive.py` — single-pass ODB extraction of U/UR into a compact numeric archive.
- `builtup_reference_section.py` — canonical four-piece geometry, walls, corners, seam-pair metadata; never cross-gap ties.
- `abaqus_modal_harmonics.py` — DOF-aware S-S longitudinal harmonic decomposition.
- `fcfsm_reference_basis.py` — K0, J_GD, equilibrium/J_D, L/D/G bases, nullspace-safe solves, cache.
- `assembly_projector.py` — common-motion, relative-piece, seam opening/sliding/slip diagnostics.
- `mechanical_modal_classifier.py` — energy/vector family shares, residuals, flags, per-mode/cluster classification.
- `test_resource_policy.py`
- `test_modal_archive.py`
- `test_reference_section.py`
- `test_modal_harmonics.py`
- `test_fcfsm_reference_basis.py`
- `test_assembly_projector.py`
- `test_mechanical_modal_classifier.py`

**Modify**
- `abaqus_complete_model_m20.py` — U+UR output and aggressive solver resources.
- `abaqus_dsm_modal_audit.py` — orchestrate archive/reference/basis/classifier and retain geometric audit.
- `abaqus_modal_validation.py` — K0/eigenspace bounds integration.
- `abaqus_modal_visuals.py` — annotate mechanical result and diagnostics on the existing mode-section plots.
- `test_complete_pipeline.py`
- `test_dsm_modal_audit.py`
- `test_modal_validation.py`
- `test_modal_visuals.py`
- `README_dsm_modal_audit.md`
- `README.md`

---

### Task 1: Progress Engine, Aggressive Resource Policy, and Abaqus Job Resources

**Files:**
- Create: `abaqus_progress.py`
- Create: `test_progress.py`
- Create: `abaqus_resource_policy.py`
- Create: `test_resource_policy.py`
- Modify: `abaqus_complete_model_m20.py`
- Modify: `test_complete_pipeline.py`

**Interfaces:**
- Produces: `ProgressTracker(stage_names, stage_weights, emit=print)`
- `ProgressTracker.start(stage)`, `update(done=None,total=None,estimate_fraction=None,note=None)`, `finish(stage)`, `summary()`
- Produces: `detect_resources() -> dict`
- Produces: `resolve_resource_plan(requested_cpus=None, requested_gpus=None, work_items=None) -> dict`
- Produces: `worker_layout(logical_cpus: int, work_items: int) -> dict`
- Produces: `gpu_array_backend() -> dict` with keys `name`, `module`, `device_count`
- Consumed later by Tasks 2, 6, and 7.

- [ ] **Step 1: Write failing progress and resource-policy tests**

Add progress tests asserting:
- stage transitions are monotonic;
- exact loops report `done/total`, percentage, rate and ETA;
- remaining-stage names are reported;
- estimated solver progress is visibly labeled `ESTIMATED`;
- an estimated stage never reaches 100% before `finish()`;
- concurrent worker updates are aggregated by the coordinator rather than printed per worker.



Add tests asserting:
- auto mode uses `os.cpu_count()` logical CPUs;
- a positive explicit CPU value is honored;
- `worker_layout` uses all logical CPU capacity in aggregate without nested oversubscription;
- aggressive memory request is 100 percent, not fixed 24000 MB;
- missing `nvidia-smi` / CuPy falls back to CPU without failure;
- a mocked supported GPU backend reports all detected GPUs.

- [ ] **Step 2: Run the tests and confirm failure**

Run:
`python -m unittest test_progress.py test_resource_policy.py test_complete_pipeline.py -v`

Expected: new tests fail because the module/resource semantics do not exist.

- [ ] **Step 3: Implement `abaqus_progress.py` and `abaqus_resource_policy.py`**

The progress engine must support exact counters and clearly labeled estimates, stage-weighted overall percent, elapsed/rate/ETA calculation, remaining-stage reporting, and persistent timing history keyed by run signature when supplied.



Use runtime capability detection only. `worker_layout` must choose process count and per-process BLAS thread count such that their product targets all logical CPUs. Do not reserve idle cores.

GPU detection is optional/capability-driven. Never require CuPy for CPU-only systems.

- [ ] **Step 4: Update the Abaqus builder resource and progress semantics**

In `abaqus_complete_model_m20.py`, wire the progress tracker through build, solver monitoring, postprocessing and modal-audit orchestration. Deterministic stages use exact counters where available; solver monitoring uses parsed trustworthy evidence first and otherwise a historical/fallback estimate that is explicitly labeled.



In `abaqus_complete_model_m20.py`:
- permit an automatic/all-core CPU setting while retaining positive integer compatibility;
- replace fixed `memory=24000, memoryUnits=MEGA_BYTES` with aggressive maximum-percentage memory configuration;
- pass all supported detected GPUs to Abaqus Job only when the Abaqus API accepts the GPU argument; otherwise record a capability fallback;
- persist resolved CPU/GPU/memory settings in `pipeline_status.json` / build report.

- [ ] **Step 5: Run regression tests**

Run:
`python -m unittest test_progress.py test_resource_policy.py test_complete_pipeline.py test_longitudinal_lines.py -v`

Expected: PASS; existing CLI forms remain accepted.

- [ ] **Step 6: Commit**

`git add abaqus_progress.py test_progress.py abaqus_resource_policy.py test_resource_policy.py abaqus_complete_model_m20.py test_complete_pipeline.py && git commit -m "feat: add aggressive resources and runtime progress"`

---

### Task 2: U+UR Modal Output and Single-Pass Numeric Archive

**Files:**
- Create: `abaqus_modal_archive.py`
- Create: `test_modal_archive.py`
- Modify: `abaqus_complete_model_m20.py`
- Modify: `abaqus_dsm_modal_audit.py`
- Modify: `test_complete_pipeline.py`
- Modify: `test_dsm_modal_audit.py`

**Interfaces:**
- Produces: `extract_modal_archive(odb, metadata, output_path: str) -> dict`
- Produces: `open_modal_archive(path: str) -> ModalArchive`
- `ModalArchive.read_mode(index) -> dict` returns `U`, `UR`, eigenvalue, mode number.
- Consumes Task 1 resource provenance.

- [ ] **Step 1: Add failing tests**

Cover:
- builder field request contains exactly `('U', 'UR')`;
- `S/E/SF/SE` remain absent;
- archive preserves mode/station/node/component order for both U and UR;
- local-coordinate nodal output is rejected;
- U-only archive/ODB produces `MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR`;
- archive metadata carries ODB/model/resource provenance.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_modal_archive.py test_complete_pipeline.py test_dsm_modal_audit.py -v`

Expected: FAIL on UR/archive requirements.

- [ ] **Step 3: Implement numeric archive extraction**

Read ODB serially once. Do not pass ODB objects into worker processes. Store compact numeric arrays and immutable geometry/label maps suitable for memory mapping/shared read-only use.

- [ ] **Step 4: Change Buckle field output to U+UR**

Modify only the Buckle classification field request; do not add shell stress/strain output.

- [ ] **Step 5: Run tests**

Run:
`python -m unittest test_modal_archive.py test_complete_pipeline.py test_dsm_modal_audit.py test_modal_visuals.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

`git add abaqus_modal_archive.py test_modal_archive.py abaqus_complete_model_m20.py abaqus_dsm_modal_audit.py test_complete_pipeline.py test_dsm_modal_audit.py && git commit -m "feat: archive U and UR mode shapes once"`

---

### Task 3: Canonical Four-Piece Reference Section

**Files:**
- Create: `builtup_reference_section.py`
- Create: `test_reference_section.py`
- Modify: `abaqus_physical_walls.py` only if a reusable corner/wall helper is needed.

**Interfaces:**
- Produces: `build_reference_section(section_segments, thickness_mm, E_MPa, nu, length_mm, seams=None) -> dict`
- Result keys: `pieces`, `nodes`, `elements`, `plate_groups`, `corner_elements`, `seam_pairs`, `material`, `definition_hash`.
- `definition_hash` must exclude bolt count, bolt positions, bolt pitch, contact and BEAM MPCs.

- [ ] **Step 1: Write failing geometry tests**

Assert:
- all four pieces exist in real global coordinates;
- no element connects nodes belonging to different pieces across a gap;
- piece numbering permutations produce the same physical-definition hash after canonical ordering;
- curved-corner elements are flagged but remain in the stiffness mesh;
- changing bolt positions/count in supplied metadata does not change the reference hash;
- changing thickness/E/geometry does change it.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_reference_section.py test_physical_walls.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement canonical geometry construction**

Build each piece independently from `section_segments`; reuse physical-wall/corner logic where correct. Never insert gap-spanning topology.

- [ ] **Step 4: Run tests**

Run:
`python -m unittest test_reference_section.py test_physical_walls.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

`git add builtup_reference_section.py test_reference_section.py abaqus_physical_walls.py && git commit -m "feat: build canonical four-piece reference section"`

---

### Task 4: fcFSM Reference Basis and K0

**Files:**
- Create: `fcfsm_reference_basis.py`
- Create: `test_fcfsm_reference_basis.py`
- Modify: `abaqus_modal_validation.py`
- Modify: `test_modal_validation.py`

**Interfaces:**
- Consumes: reference-section dict from Task 3.
- Produces: `build_fcfsm_basis(reference, harmonic_n: int, bc: str = 'S-S') -> FamilyBasis`
- `FamilyBasis` exposes `K0`, `J_GD`, `J_D`, `C_L`, `C_D`, `C_G`, `solve_k0(rhs)`, `definition_hash`, `condition_report`.
- Produces: `basis_cache_key(reference_hash, harmonic_n, bc) -> str`.

- [ ] **Step 1: Write failing analytic tests**

Cover:
- analytic diagonal `K0/J_GD/J_D` case recovers the known L/D/G force split;
- wall-force unit scaling does not alter family result;
- curved-corner elements contribute to K0 but not flat-plate tangent `J_GD`;
- no matrix row/column couples different physical pieces solely to regularize K0;
- rigid/null modes are removed/projected explicitly and reported;
- a nullspace-safe solve matches a Moore-Penrose reference on a synthetic singular system;
- cross terms between constructed L/D/G are below numerical tolerance;
- cache key is unchanged by bolt spacing/count metadata.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_fcfsm_reference_basis.py test_modal_validation.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement K0 and fcFSM operators**

Port the relevant mechanics from repository CUFSM 5.70 `analysis/fcFSM/SecAnal_fcFSM.m` / `stripmain_fcFSM.m` into focused Python, preserving units and equilibrium definitions.

Treat every written `K0^-1` as a solve on the validated energetic subspace, never as an unconditional dense inverse.

- [ ] **Step 4: Add sparse factor/cache support**

Cache basis/factorization by `definition_hash + harmonic_n + BC`. Cache data must contain no bolt-spacing input.

- [ ] **Step 5: Run tests**

Run:
`python -m unittest test_fcfsm_reference_basis.py test_modal_validation.py test_dsm_modal_audit.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

`git add fcfsm_reference_basis.py test_fcfsm_reference_basis.py abaqus_modal_validation.py test_modal_validation.py && git commit -m "feat: add whole-section fcFSM reference basis"`

---

### Task 5: Multi-Harmonic U/UR Decomposition

**Files:**
- Create: `abaqus_modal_harmonics.py`
- Create: `test_modal_harmonics.py`
- Modify: `abaqus_modal_wavelengths.py` only for shared helpers if needed.

**Interfaces:**
- Produces: `decompose_mode(z, U, UR, length_mm: float, max_harmonic: int, bc: str = 'S-S') -> dict`
- Result contains `components[m]`, `power`, `dominant_m`, `half_wavelength_mm`, `relative_residual`.
- Each harmonic component contains mapped translational and rotational cross-section amplitudes.

- [ ] **Step 1: Write failing harmonic tests**

Synthetic S-S tests:
- pure harmonic `m=1` is recovered;
- a known `m=7 + m=9` mixture recovers both amplitudes and total residual;
- sign/overall eigenvector scaling does not change normalized shares;
- irregular longitudinal stations use weighted projection correctly;
- U and UR are both represented;
- basis convention matches CUFSM S-S integral identities in `BC_I1_5.m` for equal and unequal terms.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_modal_harmonics.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement DOF-aware S-S decomposition**

Use the CUFSM S-S longitudinal convention as the reference. Keep the classifier decomposition separate from the existing display-only dominant-wavelength fit.

- [ ] **Step 4: Run tests**

Run:
`python -m unittest test_modal_harmonics.py test_modal_visuals.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

`git add abaqus_modal_harmonics.py test_modal_harmonics.py abaqus_modal_wavelengths.py && git commit -m "feat: add multi-harmonic U UR decomposition"`

---

### Task 6: Assembly and Seam-Relative Diagnostics

**Files:**
- Create: `assembly_projector.py`
- Create: `test_assembly_projector.py`

**Interfaces:**
- Consumes reference geometry from Task 3 and mapped mode amplitudes.
- Produces: `assembly_diagnostics(mode, reference, metric) -> dict`
- Produces: `seam_relative_diagnostics(mode, reference) -> dict`
- Result keys include `assembly_percent`, `normal_opening_index`, `transverse_slip_index`, `longitudinal_slip_index`, `interpiece_interaction_index`.

- [ ] **Step 1: Write failing diagnostics tests**

Assert:
- common rigid translation gives approximately zero Assembly and seam slip;
- common rigid rotation gives approximately zero false seam slip despite geometric separation;
- equal/opposite rigid motions of pieces give high Assembly;
- local plate bending inside one piece remains within-piece deformation and is not redefined as Assembly;
- imposed normal opening, transverse sliding, and longitudinal slip activate only their intended diagnostic;
- diagnostic output is invariant to overall eigenvector sign and scale.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_assembly_projector.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement common-motion and piece-motion projections**

Use a dimensionally consistent shell kinematic metric. Do not subtract this component before family classification.

- [ ] **Step 4: Implement seam diagnostics**

Use geometric seam pairing from the reference section and remove common rigid motion before evaluating gap-relative measures.

- [ ] **Step 5: Run tests**

Run:
`python -m unittest test_assembly_projector.py test_modal_visuals.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

`git add assembly_projector.py test_assembly_projector.py && git commit -m "feat: add assembly and seam diagnostics"`

---

### Task 7: Mechanical Modal Classifier with Aggressive Parallel Backend

**Files:**
- Create: `mechanical_modal_classifier.py`
- Create: `test_mechanical_modal_classifier.py`
- Modify: `abaqus_modal_validation.py`
- Modify: `test_modal_validation.py`

**Interfaces:**
- Consumes Tasks 1, 4, 5, 6.
- Produces: `classify_mode(mode_record, harmonic_result, basis_provider, diagnostics, settings) -> dict`
- Produces: `classify_modes_parallel(records, settings, resource_plan) -> list[dict]`
- Produces: `classify_eigenspace(cluster_rows, component_columns, settings) -> dict`.

- [ ] **Step 1: Write failing classification tests**

Cover:
- pure synthetic L, D, G vectors return the correct family at the 90% dominance threshold;
- a known mixture below 90% dominance returns `MIXED`;
- a known mixture returns exact K0-energy shares;
- K0-energy is primary when vector shares differ, while a >10 percentage-point discrepancy sets `METRIC_SENSITIVE`;
- mechanical residual >5% produces `UNRESOLVED`;
- harmonic reconstruction residual >5% produces `HARMONIC_FIT_POOR` and `UNRESOLVED`;
- Assembly >15% sets `HIGH_ASSEMBLY`; Assembly >25% additionally produces `UNRESOLVED`, without deleting/renormalizing the original mode before L/D/G;
- seam/inter-piece diagnostics set their flags without automatically reassigning the L/D/G family;
- rotated/scaled repeated-mode eigenspaces give identical family bounds and a non-isolated cluster cannot be certified;
- serial and all-CPU parallel outputs are bitwise or tight-tolerance equivalent;
- mocked GPU backend matches CPU reference before being accepted.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_mechanical_modal_classifier.py test_modal_validation.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement per-mode family projection**

Accumulate L/D/G/O energy across retained harmonics, report signed cross terms, kinematic cross-check shares, residual, condition data, and flags.

- [ ] **Step 4: Implement aggressive parallel scheduler**

After ODB extraction, parallelize independent mode classification using Task 1 worker/BLAS layout. Use read-only shared/memory-mapped arrays to avoid per-worker duplication.

Do not serialize mode work unless a dependency or backend thread-safety constraint requires it.

- [ ] **Step 5: Add optional GPU array backend**

Use GPU only for supported batched numeric kernels and only after a CPU-equivalence check. Record fallback reason when GPU is absent/unsupported.

- [ ] **Step 6: Implement eigenspace classification**

Reuse/extend `component_bounds`, `gram_whitener`, and spectral isolation logic from `abaqus_modal_validation.py`.

- [ ] **Step 7: Run tests**

Run:
`python -m unittest test_mechanical_modal_classifier.py test_modal_validation.py test_dsm_modal_audit.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

`git add mechanical_modal_classifier.py test_mechanical_modal_classifier.py abaqus_modal_validation.py test_modal_validation.py && git commit -m "feat: classify Abaqus modes with fcFSM mechanics"`

---

### Task 8: Integrate Mechanical Classification into --modal-audit

**Files:**
- Modify: `abaqus_dsm_modal_audit.py`
- Modify: `test_dsm_modal_audit.py`
- Modify: `test_complete_pipeline.py`

**Interfaces:**
- Consumes Tasks 1–7.
- Produces existing audit output directory plus mechanical fields in `modal_percentages.csv`, `modal_audit.json`, and summary.
- Existing `process(args)` remains the orchestration entry point.

- [ ] **Step 1: Write failing pipeline/audit tests**

Assert:
- `--modal-audit` automatically chooses mechanical classification when U+UR are available;
- U-only legacy input returns geometric report plus explicit mechanical-unavailable state;
- K0/reference hash is reused across identical geometry with different bolt metadata;
- current geometric classifier output is retained as secondary screening columns;
- all new CSV/JSON fields from the spec are emitted;
- `--resume-post ... --modal-audit` uses the existing ODB and never rebuilds/resubmits.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_dsm_modal_audit.py test_complete_pipeline.py -v`

Expected: FAIL.

- [ ] **Step 3: Replace audit orchestration path**

Sequence:
1. verify ODB/provenance;
2. extract/archive U+UR once;
3. construct/load canonical reference and basis cache;
4. harmonic-decompose;
5. run aggressive parallel mechanical classification;
6. calculate Assembly/seam diagnostics;
7. perform eigenspace/mesh checks;
8. retain geometric screening;
9. write reports.

- [ ] **Step 4: Preserve candidate/DSM evidence gates**

Do not multiply eigenvalues by modal percentages. Do not accept family critical stresses unless the existing comparison/reference/review gates pass.

- [ ] **Step 5: Run tests**

Run:
`python -m unittest test_dsm_modal_audit.py test_complete_pipeline.py test_modal_validation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

`git add abaqus_dsm_modal_audit.py test_dsm_modal_audit.py test_complete_pipeline.py && git commit -m "feat: integrate mechanical classifier into modal audit"`

---

### Task 9: Plots and Reports Matching the Piece-Wise Mode Example

**Files:**
- Modify: `abaqus_modal_visuals.py`
- Modify: `test_modal_visuals.py`

**Interfaces:**
- Consumes mechanical audit rows.
- Existing peak-section plotting remains, with separate annotations for mechanical classification, geometric screening, Assembly, seam diagnostics, and quality flags.

- [ ] **Step 1: Write failing plot-payload tests**

Assert plot title/payload contains:
- mode number and peak z;
- mechanical family and K0-energy L/D/G/O shares;
- geometric-screening family separately;
- Assembly percent separately;
- key seam/quality flag;
- original gray geometry and independently colored pieces remain available.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_modal_visuals.py -v`

Expected: FAIL.

- [ ] **Step 3: Update visual/report payloads**

Do not visually imply `Assembly + L + D + G = 100%`. Keep diagnostics on separate lines/labels.

- [ ] **Step 4: Run tests**

Run:
`python -m unittest test_modal_visuals.py test_dsm_modal_audit.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

`git add abaqus_modal_visuals.py test_modal_visuals.py && git commit -m "feat: visualize mechanical modal classification"`

---

### Task 10: Scientific Validation and CUFSM/fcFSM Benchmark Harness

**Files:**
- Create: `verify_fcfsm_classifier_benchmark.py`
- Modify: `verify_force_projector_benchmark.py`
- Create: `test_fcfsm_benchmark.py`
- Modify: `README_modal_validation.md`

**Interfaces:**
- Produces: `run_synthetic_benchmarks() -> dict`
- Produces: `compare_with_cufsm_reference(reference_json, classifier_json) -> dict`.

- [ ] **Step 1: Add failing benchmark tests**

Include:
- analytic L/D/G synthetic system;
- open-section stored reference fixture format;
- family-share and subspace-angle comparison;
- rejected benchmark when geometry/material/BC/harmonic metadata do not match.

- [ ] **Step 2: Run tests and confirm failure**

Run:
`python -m unittest test_fcfsm_benchmark.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement benchmark harness**

The CUFSM reference file must record source version, geometry, material, BC, harmonic and family results. Do not silently compare incompatible cases.

- [ ] **Step 4: Document the manual CUFSM 5.70 export/check procedure**

Document how to generate the independent open-section fcFSM reference using the repository MATLAB source. A missing MATLAB runtime is a skipped external benchmark, not a fabricated pass.

- [ ] **Step 5: Run tests**

Run:
`python -m unittest test_fcfsm_benchmark.py test_modal_validation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

`git add verify_fcfsm_classifier_benchmark.py verify_force_projector_benchmark.py test_fcfsm_benchmark.py README_modal_validation.md && git commit -m "test: add fcFSM classifier validation harness"`

---

### Task 11: Documentation, Full Regression, and Performance Evidence

**Files:**
- Modify: `README_dsm_modal_audit.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-10-02-abaqus-builtup-fcfsm-modal-classification-design.md` only if implementation terminology requires a non-semantic clarification.

**Interfaces:**
- No new code interfaces.

- [ ] **Step 1: Update user-facing command documentation**

Document:
- U+UR requirement;
- automatic aggressive resource mode;
- all-core CPU recommendation/automatic setting;
- supported GPU auto-detection and fallback;
- 100%-memory request semantics;
- mechanical vs geometric classification;
- Assembly/seam diagnostics;
- legacy U-only behavior;
- basis/cache provenance.

- [ ] **Step 2: Run the complete Python regression suite**

Run:
`python -m unittest discover -p "test_*.py" -v`

Expected: all tests PASS; Abaqus-required integration tests remain clearly separated if they need the CAE kernel.

- [ ] **Step 3: Run static compatibility checks**

Verify:
- `abaqus_complete_model_m20.py --build-only` path remains solver-free;
- old `--cpus 8` command remains accepted;
- automatic/all-core mode is accepted;
- `--buckle-output detailed` remains a compatibility no-op for heavy fields;
- Step 4/Step 5 scripts are unmodified or their tests still pass.

- [ ] **Step 4: Run one Abaqus build-only integration on the target Windows/Abaqus 2024 environment**

Expected:
- CAE + INP produced;
- no solver submission;
- resource provenance written;
- field output request in generated input/model requests U+UR only for modal classification.

- [ ] **Step 5: Run one full modal-audit integration**

Record wall-clock stages and actual CPU/GPU/memory resource provenance. Confirm the post-classification stage uses all configured CPU capacity and supported GPUs without changing results versus the serial CPU reference.

- [ ] **Step 6: Commit**

`git add README.md README_dsm_modal_audit.md docs/superpowers/specs/2026-10-02-abaqus-builtup-fcfsm-modal-classification-design.md && git commit -m "docs: document mechanical modal classification workflow"`

---

## Final Branch Review Gate

After all tasks:

- run `python -m unittest discover -p "test_*.py" -v`;
- inspect the full branch diff for accidental Step 4/Step 5 changes;
- verify no continuous cross-gap constraint entered the reference basis;
- verify K0/family hash is invariant to bolt spacing;
- verify CPU-serial and aggressive-parallel mechanical results agree;
- verify any GPU-accepted result passed CPU equivalence;
- verify new Abaqus runs request U+UR but not S/E/SF/SE;
- verify `mode_sections.png`-style outputs distinguish mechanical family from Assembly/seam diagnostics.
