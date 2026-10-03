# Built-up Abaqus modal classification: mFSM transition specification
Date: 2026-10-04 (Asia/Tehran)
Repository: azimialireza188-a11y/Alireza-1
Inspected default-branch tree: 69b0e2f9b7b61a62c51e17d3c4e5db900456a6b3
Status: Scientific specification approved by the user on 2026-10-04; implementation plan pending review.
Proposed repository destination after review: docs/superpowers/specs/2026-10-04-builtup-mfsm-modal-classification-design.md

## 1. Decision and scope

Use the Khezri–Rasmussen built-up mFSM formulation as the proposed primary mechanical classification method, subject to reproducing its basis construction and passing the validation criteria below. Retain cFSM/fcFSM/CUFSM as separately identified checks, not an automatic certificate for the four-piece member.

The scope is decomposition of existing Abaqus buckling eigenvectors of the actual four-piece model. Preserve original eigenvalues, member geometry, loads, physical material, contact definition and bolt constraints. Do not derive new critical loads from participation percentages. Restricted-family eigenvalue solutions and DSM resistance calculations are a separate subsequent task.

The target user outcome is trustworthy LOCAL/DISTORTIONAL/GLOBAL interpretation, retained participation shares, optional mechanically supported global subtype, and separate assembly/seam diagnostics. A visually local-looking mode is a regression case, not an imposed target label.

## 2. Findings in the current repository

The 2026-10-02 design and plan prescribe a bolt-independent K0 and whole-section force-based GD/D/G construction. They describe planned architecture, not an implemented automatic basis generator.

abaqus_dsm_modal_audit.py currently defaults to a geometric SCREENING PROXY. Its MechanicalProjector consumes supplied L/D/G bases and positive diagonal weights, solves joint weighted least squares, and reports component norm shares. Optional compatible K provides self energies and signed cross terms. This is not the built-up mFSM generalized eigenproblem.

abaqus_modal_shell_energy.py explicitly reconstructs physical shell membrane/bending/shear diagnostics; those quantities are not L/D/G participation.

abaqus_complete_model_m20.py builds discrete BEAM_MPC constraints and S4R shells. The automatic ModeShapes request currently stores U only. The builder also records frictionless hard general contact with contact state fixed during buckling. This physical system differs from the elastic Cartesian fasteners used in the paper's Abaqus comparison. Neither stiffness nor contact may silently be removed or changed to match the paper.

## 3. Source-grounded mathematical correction

Khezri and Rasmussen, SSRC 2023, Section 4, Eqs. (24), (29), (33)–(35), provide:
- K_shell: constituent-section elastic stiffness assembled without artificial connections across gaps;
- K_cnts: discrete connection stiffness;
- TK = K_shell + K_cnts;
- K_M: selected constituent strain-component stiffness for defining a modal space, excluding fastener stiffness.

For a compatible search basis H_M, solve

(H_M^T K_M H_M) theta = rho (H_M^T TK H_M) theta.

Select the appropriate zero-energy-ratio vectors, with the additional transverse equilibrium and orthogonality construction prescribed for each modal space. The energy ratio rho defines bases; it is NOT a modal participation percentage.

The 2023 article explicitly refers to the detailed construction in the 2018/2019 mFSM papers. Before implementing that construction, verify the mode ordering, equilibrium treatment, nullspaces and extraction against those derivations. An arbitrary strain-energy split or a renamed fcFSM projector must never be presented as the published method.

The mechanical criteria can remain fixed across connection layouts. The system metric, compatible search space and numerically constructed bases can depend on fastener layout. Therefore the previous demand that the complete family-basis hash exclude bolt pitch/count is withdrawn. Cache invariant shell operators separately from system-dependent quantities.

Source: Khezri_et_al_SSRC_2023.pdf, pp. 8–11 and 14–17; supplied file inspected directly. Official source URL: https://cloud.aisc.org/SSRC/2023/Khezri_et_al_SSRC_2023.pdf

## 4. Current rigid-constraint connection model

An ideal MPC is a kinematic constraint; it does not supply a finite bolt elastic stiffness matrix. Do not invent K_cnts, replace the MPC by an arbitrary very stiff spring, or claim measured bolt strain energy.

Construct the exact supported constraint operator C from the actual INP/connection semantics, accounting for offsets, rotations, boundary conditions and the active linearized base state. Let Q span its admissible nullspace. Work on reduced coordinates:
- K_reduced = Q^T K_shell Q;
- K_M,reduced = Q^T K_M Q;
- compatible mapping of each observed eigenvector into the same reduced space.

If contact supplies a finite tangent contribution, keep it explicitly distinguished from K_shell and K_cnts in the denominator. If the active tangent cannot be reconstructed consistently, the primary classifier is unavailable for that model; do not quietly use a contact-free metric.

This exact-constraint treatment is a proposed adaptation to the current model, not a formula established by the 2023 paper. Validate it against convergence of corresponding finite-stiffness connections toward the rigid limit, using equivalent constraint kinematics. Failure of convergence or ill-conditioning prevents activation. Do not alter the actual production Abaqus model to make this test pass.

Any unsupported MPC/contact/element configuration returns a structured UNAVAILABLE status with a reason.

## 5. Constitutive convention and shell operators

The paper's pure GBT-compatible decomposition requires suppressing Poisson coupling (p. 11) and uses nu=0 for its modal decomposition examples. Preserve physical nu in the original Abaqus eigenvalue model. Define and label an auxiliary classification operator with nu_class=0 and consistent auxiliary shear modulus; never label its energies physical modal energies of the original nu model.

Retain physical-material energy diagnostics separately if available. Record both constitutive definitions and check sensitivity. A serious disagreement is reported and inhibits scientific acceptance.

Assemble strain-component operators consistently with the reference element interpolation and local axes. Distinguish transverse coordinate x and longitudinal coordinate y from Abaqus/global axis names. Include or explicitly audit S4R transverse shear, drilling/stabilization and numerical energy differences; a thin-strip operator is not automatically identical to S4R stiffness.

Use a documented faceted approximation for curved portions, preserve actual physical gaps, and refine until classification converges. Curved segments may contribute to shell stiffness but must not be assigned flat-wall mechanical constraints without a justified mapping.

## 6. Basis construction and mapping

A single observed mode or truncated set of low buckling modes is not assumed to span the full admissible displacement space. Construct a verified search space, test its rank, and enlarge it until participation converges. Report unresolved span separately.

Use multiple longitudinal terms with displacement/rotation-appropriate basis functions. A discrete-fastener member generally couples harmonics; retain cross-harmonic system terms. Do not independently classify each harmonic with a bolt-free block and claim equivalence to Eq. (35).

Map U and UR into a canonical reference representation with a documented DOF ordering, orientation, interpolation and inverse/reconstruction check. Legacy U-only output remains screening unless a separate validated rotation-recovery scheme is later introduced. No zero-filled missing rotations.

Handle rigid/null modes using explicit energetic-space reduction, not hidden pinning or gap-spanning ties. Record rank tolerances, conditioning, removed nullspace dimension and constraint residuals.

## 7. Modal spaces and reported states

Keep mechanical L/D/G definitions and the additional shear/transverse-extension spaces needed for completeness. Do not force every vector into L/D/G. Retain any axial/global extension or other auxiliary spaces internally, but do not advertise axial extension as global buckling.

Report separately:
- dominant_family: LOCAL, DISTORTIONAL or GLOBAL when supported; null otherwise;
- quality_state: RESOLVED, MIXED, UNRESOLVED or UNAVAILABLE;
- global_subtype: FLEXURAL, TORSIONAL, FLEXURAL_TORSIONAL or null;
- L/D/G, shear, transverse-extension and residual shares, with their metric and denominator;
- assembly, seam opening and slip diagnostics with separate denominators;
- source_method and validation status.

Mixed is a mixture status. Assembly is a diagnostic. Shear and transverse extension are mechanical decomposition spaces but are not added as new DSM limit states.

Identify global subtypes using mechanically defined bending/torsion/warping content after global decomposition, not the first mode number, wavelength, or an image. In repeated eigenspaces a unique subtype may be indeterminate. Retain that indeterminacy.

## 8. Energy projection and eigenspaces

Project in the validated system classification metric, not diagonal nodal weights relabeled as strain energy. For reconstructed component u_f, self energy is 0.5 u_f^T TK_class u_f on the supported reduced space. Check all signed cross terms and reconstruction closure. Normalize to the documented total input energy, retaining residual and non-L/D/G content. Do not renormalize a failed decomposition into apparently pure L/D/G.

For near-repeated eigenvalues, analyze the full observed cluster with metric-orthonormalized vectors. Report cluster trace participation and bounds invariant under rotations of its basis. Incomplete spectral boundary clusters cannot support a definitive family minimum.

Existing numerical QC defaults (dominance 90%, residual 5%, cluster tolerance 0.1%) remain configurable engineering choices. They are not provisions of AISI S100. Assembly amplitude alone does not reassign a mechanical family.

## 9. AISI/DSM boundary

L/D/G naming is intended to support DSM interpretation. It does not establish AISI S100-2024 applicability or compliance for this built-up section.

No Fcrl/Fcrd/Fcre is automatically accepted merely because one observed eigenmode has a dominant label. Retain model/reference-stress checks, converged modes/eigenspaces, mode coverage, independent benchmark and evidence review. Missing family candidates remain unavailable, not zero and not evidence of family absence.

The supplied AISI file was located but its extracted text is image-only. Precise 2024 clause interpretation and built-up DSM applicability require rendered-page verification before publishing any compliance claim. This specification makes no such claim.

## 10. Proposed responsibilities and compatibility

- builtup_reference_section.py: canonical constituent geometry, physical walls/corners, DOF topology.
- shell_strain_operators.py: documented auxiliary shell strain-component and total operators.
- builtup_connection_operator.py: finite elastic fasteners or supported exact MPC reduction, separately recorded contact tangent.
- mfsm_reference_basis.py: source-faithful modal hierarchy, equilibrium, nullspace and metric orthogonality.
- abaqus_modal_harmonics.py: compatible multi-term U/UR mapping with coupled harmonics.
- mechanical_modal_classifier.py: system-energy participation, complete-space residuals and source/validation status.
- assembly_projector.py: independent assembly/seam diagnostics.
- existing audit/validation/visuals: integrate outputs and cluster evidence.
- builder: request U+UR for new mechanical-classification runs.

Preserve the existing CLI, build-only/resume behavior, original CAE/INP generation and original eigenvalue data. Preserve geometric screening and external-basis modes with explicit method identifiers. Old reports remain readable; do not overwrite archived results silently.

Shell-only geometry/material operators may be shared across layouts. System-dependent bases/cache keys include constraints, fastener locations/stiffness, active contact state, harmonic truncation, constitutive convention and algorithm version.

## 11. Validation and activation criteria

1. Source fidelity: reproduce the 2019 modal-space construction and 2023 channel/built-up benchmark definitions; record any deviations.
2. Algebra: symmetry, energetic-space definiteness, zero-strain criteria, equilibrium, rank, orthogonality, constraint satisfaction, signed energy closure and reconstruction.
3. Invariance: amplitude/sign, within-family basis changes and repeated-eigenspace rotations must not change accepted shares beyond numerical roundoff.
4. Published benchmarks: reproduce the paper's C2.0-90-75 and two-channel finite-fastener cases with matched geometry/material/BCs. Compare eigenvalues and subspaces, not only screenshots.
5. Rigid adaptation: demonstrate finite-fastener convergence to equivalent exact MPC constraints; reject arbitrary penalties.
6. Actual four-piece case: refine transverse reference mesh, Abaqus mesh, harmonic space and observed spectral coverage. Perform the bolt-spacing study with fixed mechanical criteria and explicitly changing system operators.
7. Cross-check: cFSM/CUFSM comparisons only when domain, constraints, material, topology and mode definitions are equivalent. Investigate differences rather than asserting one method is correct by default.
8. Regression: build-only still writes CAE+INP; historical screening runs retain their method status; U-only data never gains mechanical eligibility.
9. Performance: benchmark separately from accuracy; cache only compatible operators and bound concurrency by available memory.
10. Activation: primary default switches only after the supported physical model passes the scientific benchmarks and end-to-end Abaqus checks. A NumPy unit suite alone cannot establish this.

For research values, target at most 0.5% change in converged eigenvalues and 1 percentage point in cluster family shares between successive refinements, with shape/subspace matching. These are proposed engineering targets, not code requirements; unresolved failures block publication eligibility. Mathematical identity tests use scale-aware roundoff tolerances documented with the solver.

## 12. Review decision

Recommended: approve this corrected scientific specification as the basis for an implementation plan. Do not approve an immediate unvalidated replacement of the default classifier.

Alternative 1: retain existing screening as default while developing the mFSM backend; lowest migration risk, later activation.
Alternative 2: develop finite-fastener benchmark support first and defer exact-MPC adaptation; quickest source-faithful prototype, not sufficient for the current production connection model.
Recommended approach combines source-faithful finite-fastener verification with exact-MPC adaptation before activating mFSM on this repository's actual model.

After written-spec approval, prepare a concrete implementation plan identifying source derivation checks, test fixtures, file changes, migration and execution method. No production implementation has been performed at this review stage.


## 13. User-approved aggressive resource policy

The user explicitly requires maximum aggressive use of CPU/GPU/memory and minimum elapsed run time, with no capacity reserved for savings or other work.

- Detect all visible logical CPUs, physical cores, GPU devices and actual available memory at runtime. Remove the fixed 8-CPU default and 24000-MB Abaqus allocation.
- Default CPU request is all visible logical CPUs. Preserve explicit user CPU overrides. Benchmark process/thread topology using the full available CPU budget; select the fastest valid configuration. Do not apply a historical 12-worker cap.
- Default Abaqus memory request is 100 percent through its supported API, with getMemoryFromAnalysis disabled so a data-check recommendation cannot silently reinstate a lower ceiling. This is an allocation ceiling, not a requirement to fill RAM.
- Reserve zero bytes/percent of otherwise available RAM or VRAM. Bound allocations by live available capacity and measured working-set requirements; use no arbitrary 70/80/90 percent resource ceiling or fixed safety cushion.
- Batch multiple modes and right-hand sides, reuse factorizations, share read-only arrays, avoid repeated ODB reads and device transfers. Keep workers fed by a dynamic queue.
- Detect compatible GPU compute capability and use measured FP64 kernels when supported and faster, distributing independent batches across usable GPUs. Never claim GPU acceleration for a phase that does not support it, change the buckling procedure for GPU support, or introduce lower precision to inflate throughput.
- Changing worker count, batch shape, backend or cache reuse must preserve scientific tolerances. Do not loosen 0.5 percent eigenvalue and 1 percentage-point participation convergence targets for speed.
- OOM handling reduces an oversized allocation only to make it fit actual capacity; it is not a standing reserve policy. Diagnose unsupported GPUs/drivers/Abaqus APIs and record CPU fallback.
- Emit hardware inventory, allocation requests, selected topology/backend, per-phase elapsed time, observed peak memory and cache statistics. Hardware utilization is evidence; minimum wall time is the optimization objective.
- Abaqus installation/version/license and GPU runtime capability are external execution constraints, not artificial code caps. Record limitations encountered rather than claiming guaranteed 100 percent utilization or an unmeasured speedup.
