# Built-up mFSM Classification and Aggressive Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement scientifically auditable built-up mFSM classification of Abaqus modes with aggressive use of all available compute resources and measured minimum elapsed time.

**Architecture:** Keep source-faithful strain-component basis construction separate from actual connection/contact handling, mode mapping and reporting. Integrate a full-resource scheduler around batch matrix operations and one-pass ODB extraction. Switch the primary classifier only after scientific validation of the supported model.

**Tech Stack:** Abaqus 2024 Python, NumPy; SciPy sparse algebra and threadpoolctl in a compatible compute interpreter; optional CuPy/CUDA FP64 where supported. Preserve the existing Abaqus command workflow; capability-check dependencies instead of assuming CUDA packages are importable in Abaqus.

**Spec:** ../specs/2026-10-04-builtup-mfsm-modal-classification-design.md

**Status:** Ready for user plan review. Scientific specification approved; implementation has not started.
**Execution recommendation:** Native implementation in this session. Tasks share mathematical operators and schemas, so a single implementer minimizes repeated loading and interface drift. Independent final review follows implementation.
**Inspected main commit:** 69b0e2f9b7b61a62c51e17d3c4e5db900456a6b3

## Global Constraints

- Preserve physical Abaqus geometry, loads, material, contact, discrete BEAM_MPC constraints and eigenvalues.
- No artificial compatibility across physical gaps and no fictitious finite MPC bolt stiffness.
- K_M excludes fastener stiffness; the system denominator includes supported connection/contact effects.
- Auxiliary classification nu_class=0 is labeled separately from the physical material.
- Do not manufacture a complete basis from a single/truncated observed eigenvector set.
- No missing UR zero filling; legacy U-only ODBs stay explicitly geometric screening.
- Report LOCAL/DISTORTIONAL/GLOBAL separately from MIXED/UNRESOLVED/UNAVAILABLE and assembly diagnostics.
- No automatic DSM acceptance from a dominant participation label.
- Dominance=90%, residual=5%, cluster tolerance=0.1% are configurable numerical QC, not AISI provisions.
- Convergence targets: eigenvalue change <=0.5%; cluster share change <=1 percentage point, with subspace matching.
- No fixed CPU worker cap, RAM/VRAM reserve, or conservative resource-percentage ceiling. Allocation decisions reflect live available capacity and measured requirements.
- Default CPU request: all visible logical CPUs; Abaqus memory request: 100 percent; physical overrides remain explicit.
- FP64 on CPU/GPU; equal scientific tolerances across backends.
- Preserve build-only CAE+INP generation and all existing CLI modes; retain old report readers.
- Never claim validation or a speedup without observed evidence from the relevant hardware/model.

## Review Focus

1. Missing UR, mismatched DOF maps or incomplete spectral clusters must not acquire mechanical eligibility (Tasks 5 and 8).
2. General contact with unknown active tangent and unsupported MPC forms must return an explicit unavailable reason (Task 4).
3. Symmetric repeated modes, eigenvector rotations and incomplete search spaces must not yield unstable confident labels (Tasks 6 and 7).
4. Windows spawn, unavailable CUDA, multiple GPUs and allocation failure must preserve deterministic outputs and zero-reserve policy (Tasks 2 and 7).
5. Concurrent layout caches or interrupted reports must not reuse incompatible connection bases or overwrite historical outputs (Tasks 6 and 8).

---

## File and interface map

- `mfsm_model.py`: versioned dataclasses for ReferenceSection, OperatorPack, BasisPack, ModeBatch, ClassificationResult, ResourceInventory and ResourcePolicy.
- `runtime_resources.py`: CPU/memory/GPU detection, zero-reserve policy and backend selection.
- `modal_batch_executor.py`: scheduling/shared-array lifecycle, streaming numerical batches and telemetry.
- `builtup_reference_section.py`: canonical geometry/physical topology.
- `shell_strain_operators.py`: physical/auxiliary component operators.
- `builtup_connection_operator.py`: elastic fasteners, supported rigid reduction and contact requirements.
- `abaqus_modal_harmonics.py`: U/UR extraction, compatible mapping and cross-harmonic representation.
- `mfsm_reference_basis.py`: prescribed modal hierarchy and system-aware cache.
- `mechanical_modal_classifier.py`: energy projection, residuals and eigenspace bounds.
- `assembly_projector.py`: independent piece/seam motion diagnostics.
- Existing builder/audit/validation/visuals integrate these modules.
- New pure-Python tests run outside Abaqus; Abaqus-specific tests are separately labeled.

## Task 1: Lock source derivation and data contracts

**Files:** Create `mfsm_model.py`, `docs/mfsm-source-map.md`, `tests/test_mfsm_model.py`, `tests/fixtures/mfsm/source_manifest.json`.

**Interfaces:** Dataclasses validate shapes, FP64, units, DOF order, independent physical/classification material definitions and source hashes. OperatorPack carries shell/system/component matrices and admissible coordinates; BasisPack carries family/subspace bases plus metric and validation provenance.

- [ ] Write tests rejecting mismatched DOF counts, missing source identity and an asserted validated basis without benchmark evidence.
- [ ] Run `python -m pytest tests/test_mfsm_model.py -q`; verify the new interfaces initially fail to import.
- [ ] Read the full 2019 Part 1/Part 2 derivations and map each implemented subspace, ordering, equilibrium operation and torsion rule to its source equation/page. Add the 2023 built-up changes and identify which statements are project adaptations. Record unavailable source text as a blocker for source-faithful implementation, not permission to invent equations.
- [ ] Implement the dataclasses and source manifest; tests pass.
- [ ] Commit the contracts/source map.

## Task 2: Aggressive runtime policy and builder allocation

**Files:** Create `runtime_resources.py`, `tests/test_runtime_resources.py`; modify `abaqus_complete_model_m20.py`, `test_complete_pipeline.py`.

**Interfaces:**
`detect_resources() -> ResourceInventory`
`resolve_policy(inventory, cpu_override=None, gpu_override=None) -> ResourcePolicy`
`abaqus_job_settings(policy, capabilities) -> dict`

- [ ] Test a 24-logical-CPU, 64-GiB, two-GPU inventory: request 24 CPUs, memory=100/PERCENTAGE, getMemoryFromAnalysis=False; RAM/VRAM reserve=0. Test explicit cpus=8, unknown GPU API, CPU-only host and Windows discovery.
- [ ] Run the targeted tests and observe failures before editing the builder.
- [ ] Change `--cpus` default to auto using an internal sentinel while continuing to accept existing positive integers; remove memory=24000. Add `--gpus auto|0|N` and capability-aware supported submission settings. Check both solver and postprocessing resource policies.
- [ ] Configure native BLAS thread budgets before compute imports; record scheduler/topology choices. Benchmark physical-core versus logical-core and process/thread splits under the full available resource budget; no arbitrary 12-worker ceiling.
- [ ] Verify CLI/build settings with a stubbed Abaqus Job, then existing pipeline tests; commit.

## Task 3: Reference geometry and strain-component operators

**Files:** Create `builtup_reference_section.py`, `shell_strain_operators.py`, `tests/test_shell_strain_operators.py`, `tests/test_builtup_reference_section.py`.

**Interfaces:**
`build_reference(source_inputs, discretization) -> ReferenceSection`
`assemble_operators(section, harmonic_terms, material) -> OperatorPack`

- [ ] Write tests for independent constituent topology, rigid-motion strain, analytical flat-strip patch states, operator symmetry and material split; gaps introduce no elements/ties.
- [ ] Run those tests and confirm failures.
- [ ] Implement consistent local strain/curvature operators with explicit axis/rotation conventions and full harmonic coupling support. Retain sparse matrices or matrix actions; do not densify full large shell operators by default.
- [ ] Label auxiliary nu=0 separately; audit shear/drilling/stabilization mismatch with S4R. Implement curved-segment refinement with provenance rather than visually assigned constraints.
- [ ] Verify operator energy against independent analytical patch cases and discretization refinement; commit.

## Task 4: Actual connections, constraints and contact

**Files:** Create `builtup_connection_operator.py`, `tests/test_builtup_connection_operator.py`.

**Interfaces:**
`assemble_connections(section, layout, connection_definition) -> OperatorPack`
`reduce_constraints(operators, constraint_definition, active_contact) -> OperatorPack`

- [ ] Test a finite connector that changes system stiffness but not K_M; test an offset BEAM_MPC rigid translation/rotation field; test unsupported MPC and unknown contact tangent produce UNAVAILABLE.
- [ ] Run tests before implementation.
- [ ] Derive supported BEAM_MPC reduction from verified Abaqus semantics, including rotational coupling and offsets. Implement an admissible nullspace/reduction action without artificial cross-gap constraints or finite penalty substitution.
- [ ] Keep contact tangent separate and tied to the actual base state. Require explicit inactive/contact-tangent evidence; no assumption that declared contact contributes zero.
- [ ] Test finite-stiffness convergence toward the equivalent rigid constraints on several increasing stiffness levels and verify energetic-space rank/null handling. Commit only supported behavior.

## Task 5: U/UR and multi-harmonic mapping

**Files:** Create `abaqus_modal_harmonics.py`, `tests/test_modal_harmonics.py`; modify builder output and `test_complete_pipeline.py`.

**Interfaces:**
`extract_mode_batches(odb, reference, batch_size) -> iterator[ModeBatch]`
`map_modes(batch, reference, harmonics, operators) -> ModeBatch`

- [ ] Test sine/cosine-appropriate translations/rotations, multiple harmonics, unequal meshes, missing UR and incompatible DOF ordering.
- [ ] Verify failure, then request U+UR for mechanical-capable runs and update build metadata. Keep heavy shell output optional.
- [ ] Read each required ODB field once per mode into shared/memory-mapped arrays; worker processes do numerical work without sharing live ODB handles.
- [ ] Implement compatible mapping and reconstruction checks, preserve coupled harmonic terms, and reject unknown mapping. Test rank/coverage growth rather than assuming a small mode set spans the space.
- [ ] Run targeted and builder regression tests; commit.

## Task 6: Source-faithful mFSM bases and compatible cache

**Files:** Create `mfsm_reference_basis.py`, `tests/test_mfsm_reference_basis.py`, `tests/test_mfsm_cache.py`.

**Interfaces:**
`build_mfsm_basis(operators, source_manifest, policy) -> BasisPack`
`basis_cache_key(section, operators, source_manifest) -> str`

- [ ] Test the generalized energy-ratio nullspace problem against analytically constructed tiny systems; verify equilibrium, energetic orthogonality, completeness and rigid-null handling.
- [ ] Test bolt-layout/contact/material/harmonic changes invalidate system-basis keys while unchanged shell operators may be reused. Test interrupted and concurrent cache creation.
- [ ] Implement the Task 1 verified hierarchy, including S/TE and auxiliary spaces; select zero ratios with scale-aware tolerances. Ratio eigenvalues remain separate from participation.
- [ ] Factor each compatible metric once, solve multiple right-hand sides, use reduced dense blocks only when they fit live capacity, otherwise sparse/operator solves.
- [ ] Implement atomic cache writes and provenance/hash checks. Verify basis enlargement converges; commit.

## Task 7: Energy classifier, eigenspaces and CPU/GPU batches

**Files:** Create `mechanical_modal_classifier.py`, `modal_batch_executor.py`, `tests/test_mechanical_modal_classifier.py`, `tests/test_modal_batch_executor.py`; extend `abaqus_modal_validation.py`.

**Interfaces:**
`classify_modes(batch, basis, thresholds, policy) -> list[ClassificationResult]`
`classify_cluster(batch, basis, thresholds) -> dict`
`execute_batches(batches, operation, policy) -> iterator`

- [ ] Test positive/negative amplitude, known mixed-space examples, non-L/D/G content, signed cross terms, residual failure and eigenspace rotation invariance.
- [ ] Test CPU/GPU FP64 equivalence, unsupported CUDA fallback, spawn-safe worker setup, deterministic row ordering, multi-GPU scheduling and allocation failure.
- [ ] Implement projection in TK_class, complete denominators, cross-term closure, residuals and cluster trace/bounds. A cluster truncated at the spectral boundary remains ineligible.
- [ ] Implement dynamic batching across usable devices, synchronized GPU timing, one-transfer batch residency and shared read-only CPU arrays. Compare timings including transfers before choosing CPU/GPU.
- [ ] Use live available memory minus measured simultaneous working sets, with zero arbitrary reserve. On allocation failure release failed buffers and fit smaller batches; report retry and backend.
- [ ] Run classifier/cluster/backend tests, skipping real-device comparison only with an explicit unavailable reason; commit.

## Task 8: Diagnostics and end-to-end report integration

**Files:** Create `assembly_projector.py`, `tests/test_assembly_projector.py`; modify `abaqus_dsm_modal_audit.py`, `abaqus_modal_visuals.py`, `abaqus_modal_report.py`, existing audit/report tests.

**Interfaces:**
`assembly_diagnostics(batch, section, seams) -> dict`
Audit option `--classifier auto|screening|external|mfsm`; auto selects mFSM only after supported-model validation.
Results expose dominant_family, quality_state, global_subtype, source_method, energy metric, complete-space shares, diagnostics and validation evidence.

- [ ] Test pure common rigid motion yields no false seam slip; independent piece motion remains diagnostic without removing it before classification.
- [ ] Test legacy reports/CLI, U-only ODB, unsupported physics, incomplete cluster, empty family candidates, explicit mFSM request and incompatible external pack.
- [ ] Implement mechanically supported global bending/torsion/warping subtype; unresolved subtype stays null. Retain separate shear/TE/residual.
- [ ] Integrate versioned CSV/JSON/plots; distinguish auxiliary energy, physical shell energy and vector norm metrics. Preserve existing candidates/evidence gates.
- [ ] Write reports atomically to new output directories; no historical overwrite. Add resource telemetry and cache/backend provenance.
- [ ] Run report and archive regressions; commit.

## Task 9: Published and production validation, measured performance

**Files:** Create `benchmarks/mfsm/run_validation.py`, `benchmarks/mfsm/run_performance.py`, `tests/test_mfsm_benchmark_manifest.py`; add benchmark manifests and result schema under `benchmarks/mfsm/`.

**Interfaces:** JSON results record source/model hashes, tolerances, numerical outcomes, hardware inventory, stage timing, peak memory, cache reuse and backend. Activation evidence binds to supported physical/model definition and algorithm version.

- [ ] Test that unit-only evidence, unmatched connector semantics, changed model hashes or absent Abaqus results cannot activate the primary classifier.
- [ ] Reproduce paper channel/two-channel cases using verified geometry/material/BC/fastener values; do not digitize unsupported graph precision into claimed measurements.
- [ ] Compare CPU/GPU and reference mesh/harmonic refinement at identical tolerances; verify <=0.5% eigenvalue and <=1-pp cluster-share change with matched subspaces.
- [ ] Validate current four-piece MPC/contact case against two Abaqus meshes and appropriate independent references. If the execution host lacks Abaqus or required input data, produce executable validation commands and mark evidence PENDING, never PASSED.
- [ ] Benchmark cold/warm cache, batch sizes, CPU/thread topology and all usable GPUs on representative jobs. Report actual elapsed times; no asserted minimum-time guarantee.
- [ ] Activate auto-mFSM only for configurations with passing evidence. Unsupported configurations remain explicit screening/unavailable; commit validation results and commands.

## Task 10: Documentation, independent review and direct GitHub integration

**Files:** Modify `README.md`, `README_dsm_modal_audit.md`; create `README_mfsm.md`, `README_resource_execution.md`.

- [ ] Document one-command runs, legacy/resume behavior, GPU/runtime setup and all resource/provenance fields.
- [ ] Document fixed-8-CPU/fixed-24000-MB removal, auto resources, explicit overrides and verified backend limits.
- [ ] Run `python -m pytest -q` on the supported pure-Python environment and the separately labeled Abaqus checks where available.
- [ ] Independently review mechanical definitions, source fidelity, actual MPC/contact reduction, repeated-eigenspace handling, resource policy and CLI migration; fix substantive findings.
- [ ] Review final diff and scientific evidence; push verified changes directly to the authorized repository without force-overwriting unrelated commits.
- [ ] Report commit URL, implementation/validation status, measured timing and any remaining actual Abaqus/GPU evidence limitation.

## Execution order and review decision

Tasks 1–6 establish scientifically meaningful operators/bases. Task 7 adds classification and aggressive batch execution; Task 8 connects production reports; Task 9 controls activation; Task 10 integrates the verified result.

No resource tuning changes the physical model or relaxes numerical acceptance. Resource usage is unrestricted by savings/reservation policies, while scheduler choices optimize measured elapsed time.

Required next step under Superpowers: the user reviews this written plan and selects Native or Subagent-driven execution before implementation. Native is recommended. Approval of this plan authorizes task-by-task implementation and the direct GitHub integration already requested.
