# Abaqus Built-Up fcFSM-Style Modal Classification — Design Specification

Date: 2026-10-02  
Repository: `azimialireza188-a11y/Alireza-1`  
Scope: Stage A mechanical classification of existing Abaqus eigenmodes. Stage B restricted-family eigenproblems are explicitly deferred until Stage A is validated.

## 1. Goal

Replace the current geometry-driven Local/Distortional/Global interpretation with a mechanically defined, whole-built-up-section classifier inspired by CUFSM/fcFSM.

Abaqus remains the eigenvalue solver. The classifier shall not alter Abaqus eigenvalues or invent family critical stresses. It shall decompose each observed eigenmode of the full four-piece bolted member into mechanically defined Local, Distortional, Global and residual components, while separately reporting inter-piece/connection-sensitive motion.

The target system is the complete four-piece built-up cross-section, not four separately classified lipped angles.

## 2. Physical interpretation established from the current model and mode plots

The four constituent pieces occupy distinct geometric locations and are separated by clear gaps except at discrete physical bolt links. The example peak-displacement plots show that individual pieces can undergo local motion relative to neighboring pieces between connection points.

Therefore the reference classification model MUST NOT impose artificial continuous compatibility between different pieces.

Specifically, across a gap between two separate pieces:

- no continuous translational tie is permitted;
- no continuous rotational tie is permitted;
- no artificial strip, plate or shell is inserted across the gap;
- no continuous seam stiffness is introduced;
- no assumption equivalent to `u_A = u_B`, `v_A = v_B`, or `w_A = w_B` is allowed between distinct pieces.

Within each physical piece, the shell displacement field remains continuous according to its actual finite-element or canonical reference discretization.

The actual Abaqus model continues to contain the real discrete BEAM MPC bolt links. Those links influence the real eigenvalue and eigenvector. They do not change the family definitions.

## 3. Architectural principle

The workflow is:

```text
actual Abaqus eigenmode of full bolted member
        |
        +--> longitudinal harmonic decomposition
        |
        +--> whole-section mechanical L/D/G decomposition
        |
        +--> Assembly diagnostic
        |
        +--> seam-relative-motion diagnostics
        |
        +--> eigenspace/quality checks
```

Assembly is a diagnostic, not a fourth DSM buckling family and not a preprocessing subtraction.

The classifier shall never remove an “Assembly component” from the eigenmode before L/D/G classification. Relative piece motion may coexist with a real Local or Distortional mechanism and must not be erased.

## 4. Reference stiffness K0

A bolt-spacing-independent elastic reference stiffness `K0` is used to define the mechanical metric and family bases.

`K0` depends on:

- the geometry of all four physical pieces in their real locations;
- thickness;
- elastic material properties;
- member length / longitudinal harmonic definition;
- canonical reference discretization.

`K0` does NOT depend on:

- bolt count;
- bolt pitch;
- bolt end distance;
- discrete BEAM MPC locations;
- contact state;
- friction;
- bolt preload;
- bearing;
- washer or bolt-hole details.

Conceptually, before longitudinal coupling/reduction:

```text
K0 = blockdiag(K_piece_1, K_piece_2, K_piece_3, K_piece_4)
```

with no artificial cross-gap compatibility.

This keeps the family definition invariant when bolt spacing changes. Bolt spacing changes the observed Abaqus mode, not the definition of Local/Distortional/Global.

## 5. Longitudinal representation

An Abaqus eigenmode of the discrete-bolted member may contain more than one longitudinal harmonic. Classification shall therefore use a multi-harmonic representation rather than only the dominant half-wavelength.

For a simply supported member, the mode is represented schematically as

`u(z) = sum_m u_m f_m(z) + r(z)`.

The implementation shall use DOF-appropriate longitudinal basis functions. It must not blindly fit every displacement and rotation component with the same scalar sine basis.

For each retained harmonic `m`, a compatible reference basis and `K0,m` are used. Family contributions are accumulated across retained harmonics. Harmonic reconstruction residual is reported independently.

The existing `abaqus_modal_wavelengths.py` remains useful for wavelength reporting, but the mechanical classifier gets its own longitudinal mapping layer.

## 6. Mechanical family definitions

The design follows the force-based philosophy in the repository CUFSM 5.70 fcFSM code, especially:

- `analysis/fcFSM/SecAnal_fcFSM.m`;
- `analysis/fcFSM/stripmain_fcFSM.m`.

### 6.1 Physical plates and curved corners

Flat physical walls from all four pieces participate in the generalized wall-force system. Curved-corner strips remain part of the elastic stiffness `K0`, but are excluded from the flat-plate tangent load definitions, consistent with fcFSM’s `cornerStrips` concept.

No gap is treated as a wall or plate.

### 6.2 GD space

For the complete four-piece cross-section, define generalized tangent wall forces

`q = [q_1, q_2, ..., q_n]^T`

and the equivalent generalized nodal loading

`P = J_GD q`.

Then

`GD = range(K0^-1 J_GD)`.

This is a mechanical definition and must not be inferred from apparent fold motion in plots.

### 6.3 Local

Local is the `K0`-orthogonal complement of GD:

`L = ker(J_GD^T)`.

Local is therefore not “whatever remains after Distortional” and not a visually defined wall-bending residual.

### 6.4 Distortional

Distortional uses wall-force combinations that are in equilibrium for the complete built-up cross-section.

The equilibrium conditions include whole-section in-plane force and moment equilibrium, producing a basis `J_D` for admissible distortional wall forces.

Then

`C_D = K0^-1 J_GD J_D`.

The equilibrium conditions are enforced for the complete four-piece system, not independently for each lipped-angle piece.

### 6.5 Global

Global is the remainder of GD that is `K0`-orthogonal to D:

`J_G = null(C_D^T J_GD)`

and

`C_G = K0^-1 J_GD J_G`.

Thus GD is partitioned mechanically into Global and Distortional.

### 6.6 Residual / Other

The fourth reported quantity is a mechanical reconstruction residual, not an independently asserted physical buckling family.

For the compatible reconstructed motion:

`u = u_L + u_D + u_G + u_O`.

A large `O` fraction is a quality warning such as `MECHANICAL_BASIS_RESIDUAL_HIGH`, not automatic evidence of an “Other buckling mode.”

## 7. Primary and secondary metrics

### 7.1 Primary metric: K0 strain energy

The primary family classification metric is the elastic reference energy.

For component `f`:

`E_f = 0.5 * u_f^T K0 u_f`.

The primary percentages are normalized from the mechanically reconstructed component energies, with signed cross terms also calculated and reported for orthogonality auditing.

The basis construction shall make L/D/G mutually compatible with the chosen `K0` metric. Large cross terms are a failure condition, not something silently renormalized away.

### 7.2 Secondary metric: kinematic/vector check

A dimensionally consistent kinematic metric is reported only as a cross-check.

Translation and shell rotation must not be directly combined as mm and radians. The rotational contribution shall be scaled using a shell-consistent geometric measure based on thickness / through-thickness displacement equivalence, rather than an arbitrary user-selected length.

A large change in family interpretation between the `K0` metric and the kinematic metric produces a `METRIC_SENSITIVE` flag. The `K0` metric remains primary.

## 8. Required Abaqus modal output

New buckling runs with mechanical classification enabled shall store both:

- `U`;
- `UR`.

The automatic buckling pipeline shall continue NOT to request heavy shell fields `S`, `E`, `SF`, or `SE`.

Legacy ODBs containing only `U` remain usable by the old geometric screening path. They must not be silently promoted to full mechanical classification. Missing rotations must yield an explicit status such as `MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR`.

## 9. Assembly diagnostic

Assembly is an independent diagnostic measuring relative rigid-like motion of the four constituent pieces with respect to the common overall motion of the complete built-up cross-section.

It is not:

- a DSM family;
- subtracted from the mode before L/D/G classification;
- defined as violation of an artificial continuous seam tie.

A suitable implementation fits:

1. the common rigid/low-order motion of the full four-piece cross-section;
2. the corresponding rigid-like motion of each individual piece;
3. the relative piece motions after removing the common motion.

The diagnostic reports how strongly the mode contains inter-piece relative motion.

Its denominator is explicitly documented and is independent of the normalized L/D/G/O denominator. Therefore `L + D + G + O = 100%` does not imply that Assembly must be added to that sum.

## 10. Seam-relative-motion diagnostics

Because the pieces are physically separated between bolts, the classifier shall report actual relative motion at paired seam-edge locations without treating it as a compatibility error.

At minimum the diagnostics shall resolve:

- normal opening/closing across the local gap;
- in-plane transverse sliding along the seam-edge direction;
- longitudinal slip.

These measures are computed after removing the appropriate common rigid motion so that a pure overall translation or rotation of the complete cross-section does not create false seam slip.

The seam diagnostics are descriptive. They do not automatically reassign a mode from D to Assembly or from L to Assembly.

## 11. Inter-piece force-resultant diagnostic

The classifier shall optionally report the degree to which equilibrium is achieved through opposing resultants among different pieces.

A whole-section Distortional candidate may satisfy total cross-section equilibrium while individual pieces carry large opposing resultants. This shall be recorded using a diagnostic such as `INTERPIECE_INTERACTION_HIGH`.

The diagnostic does not change the family definition by itself.

## 12. Repeated / clustered eigenvalues

The four-piece symmetric member may generate repeated or nearly repeated eigenvalues. Classification must therefore be eigenspace-safe.

For eigenvalues within the configurable cluster tolerance, the program shall analyze the full observed eigenspace and report family-share bounds over admissible linear combinations.

A family is considered stable only if the eigenspace-level result supports that conclusion. Individual eigenvector orientation from the eigensolver must not be treated as unique physics when the eigenspace is numerically degenerate.

The existing `abaqus_modal_validation.py` eigenspace machinery shall be reused and strengthened rather than replaced.

## 13. Geometric classifier status

The current geometric `SectionProjector` / curved-wall classifier is retained as a screening and visual-audit tool.

It is no longer the primary scientific basis for L/D/G assignment once the mechanical classifier is available.

The report shall allow comparison such as:

```text
Mechanical: Local 93%
Geometric screening: Local 86%
```

and explicitly flag material disagreement.

## 14. Example mode-plot behavior that the implementation must handle

The supplied example plot contains modes where the four pieces show noticeably different local motions, including cases currently labeled D, Mixed, Unresolved and Assembly.

The new classifier shall treat those plots as observed eigenmodes of one built-up system while preserving the physical freedom of the separated pieces. It shall not infer a false continuous seam merely because the four pieces form a closed-like overall box.

In particular, the implementation must remain valid when:

- one or two pieces exhibit larger local wall bending than the others;
- lips on opposite pieces move in different directions;
- relative piece rotation is visible;
- motion changes strongly between bolt locations;
- the current geometric classifier calls the mode Assembly;
- the current geometric classifier assigns very high D even though visible wall bending is present.

The supplied labels are regression examples, not ground-truth labels to hard-code.

## 15. Classification states and quality flags

Allowed final states:

- `LOCAL`;
- `DISTORTIONAL`;
- `GLOBAL`;
- `MIXED`;
- `UNRESOLVED`.

Initial configurable QC thresholds:

- dominant family: 90%;
- mechanical residual warning/failure: 5%;
- Assembly warning: 15%;
- Assembly unresolved gate: 25%;
- energy/vector share sensitivity: 10 percentage points;
- eigenvalue cluster tolerance: 0.1%;
- harmonic reconstruction residual: 5%.

These are engineering QC defaults, not code or standard limits, and must be emitted in provenance.

Example flags include:

- `HIGH_ASSEMBLY`;
- `HIGH_SEAM_RELATIVE_MOTION`;
- `INTERPIECE_INTERACTION_HIGH`;
- `HIGH_MECHANICAL_RESIDUAL`;
- `METRIC_SENSITIVE`;
- `HARMONIC_FIT_POOR`;
- `EIGENSPACE_NOT_ISOLATED`;
- `BASIS_ILL_CONDITIONED`;
- `MECHANICAL_CLASSIFICATION_UNAVAILABLE_MISSING_UR`.

## 16. Stage A outputs

For each mode, the main CSV shall include at least:

- mode number;
- eigenvalue;
- critical stress;
- dominant harmonic;
- dominant half-wavelength;
- `L_energy_percent`;
- `D_energy_percent`;
- `G_energy_percent`;
- `O_energy_percent`;
- corresponding kinematic/vector cross-check shares;
- `assembly_percent`;
- seam normal-opening index;
- seam transverse-slip index;
- seam longitudinal-slip index;
- inter-piece interaction diagnostic;
- final family;
- quality state;
- flags;
- harmonic residual;
- mechanical residual;
- metric sensitivity;
- eigenspace cluster id.

`modal_audit.json` shall additionally store all basis metadata, hashes, thresholds, matrix conditioning and provenance.

The graphical report shall continue to include peak-displacement cross-sections like the supplied example, but annotate mechanical family, geometric screening family, Assembly diagnostic and key QC flags separately.

## 17. Proposed code boundaries

The implementation shall keep responsibilities isolated:

```text
abaqus_complete_model_m20.py
    - existing builder/solver orchestration
    - request U + UR for new buckling runs
    - preserve all current CLI behavior

abaqus_dsm_modal_audit.py
    - orchestration and report assembly only

abaqus_modal_harmonics.py
    - DOF-aware longitudinal decomposition

builtup_reference_section.py
    - canonical four-piece geometry
    - physical walls/corners
    - no artificial cross-gap constraints

fcfsm_reference_basis.py
    - K0
    - J_GD
    - equilibrium operator / J_D
    - L/D/G bases
    - cache/provenance

assembly_projector.py
    - common-motion and relative-piece diagnostics
    - seam-relative-motion diagnostics

mechanical_modal_classifier.py
    - K0-energy projection
    - kinematic cross-check
    - residual/cross-term checks
    - family/quality state

abaqus_modal_validation.py
    - eigenspace and mesh-to-mesh validation
```

The exact file split may change during planning if a smaller dependency surface is demonstrably cleaner, but these responsibilities must remain isolated.

## 18. Canonical geometry and mesh independence

Family definitions shall not be learned from one particular Abaqus mesh.

A canonical reference section is constructed from the source geometry and material/thickness data. Abaqus nodal `U/UR` is mapped to that reference representation.

The same physical section analyzed with different shell mesh sizes shall use the same family definition and equivalent `K0` reference, within documented numerical discretization tolerance.

This is required for mesh convergence and for comparing bolt-count cases without moving the classifier target.

## 19. Backward compatibility

The existing one-command workflow is preserved.

Existing options such as:

- `--build-only`;
- `--resume-post`;
- `--buckle-output`;
- `--nodal-precision`;
- `--longitudinal-lines`;
- `--longitudinal-line-min-spacing-mm`;
- `--modal-audit`

must remain valid.

`--build-only` must still create CAE + INP without solver submission.

CAE and INP generation must remain intact whenever model building occurs.

The separate Step 4 imperfection and Step 5 GMNIA scripts are outside this change and must not be broken.

The historical `--buckle-output detailed` compatibility option shall remain accepted and shall not re-enable `S/E/SF/SE` automatically.

## 20. Validation strategy

Stage A is not considered scientifically usable until all of the following pass.

### 20.1 Mathematical synthetic tests

Construct synthetic vectors known to lie in L, D and G subspaces. Confirm:

- correct family recovery;
- K0-orthogonality;
- small cross terms;
- residual behavior;
- invariance to basis-column scaling;
- stable eigenspace bounds for rotated repeated-mode bases.

### 20.2 CUFSM/fcFSM open-section benchmark

Use one compatible open-section problem for which repository CUFSM/fcFSM can provide reference L/D/G results.

The new classifier must reproduce the family interpretation to an explicitly documented tolerance before the built-up case is relied upon.

### 20.3 Four-piece reference tests

Check:

- C4 symmetry invariance;
- no dependence on piece numbering;
- no dependence on arbitrary sign of eigenvectors;
- no artificial continuity across gaps;
- zero/low Assembly for common rigid motion;
- nonzero Assembly for imposed relative piece motion;
- seam diagnostic response to controlled opening/sliding/slip.

### 20.4 Mesh comparison

Compare at least two practical Abaqus meshes mapped onto the same canonical reference representation.

Family shares, eigenspace interpretation and candidate critical stresses must be checked using the existing mesh-validation framework.

### 20.5 Bolt-spacing invariance of the classifier definition

For identical section geometry/material with different bolt counts/spacings:

- `K0` hash and family-definition hash remain unchanged;
- observed Abaqus eigenvalues/modes may change;
- Assembly and seam diagnostics may change;
- L/D/G shares may change because the observed mode changes;
- the definition used to compute those shares does not change.

## 21. Regression use of known modes

Existing stored/available modes such as the previously discussed short-half-wave Local-like and longer-wave D/L-boundary cases shall be used as regression examples.

No expected family percentage is hard-coded merely to reproduce prior visual judgment.

The important regression condition is that the new mechanical method explains the mode with stable mechanics and does not recreate the previous failure mode in which coarse fold motion forced obvious wall bending into a nearly pure D label.

## 22. Stage B boundary

Stage B is intentionally not implemented as part of this design.

After Stage A is validated, the same reference subspaces may be used to formulate restricted-family eigenproblems for independently resolving `FcrL`, `FcrD` and `FcrG`.

No Stage B critical stress shall be reported before the Stage A basis and projection pass the validation sequence above.

## 23. Provenance and reproducibility

Every mechanical audit shall record hashes/identifiers for:

- source geometry;
- material and thickness;
- canonical reference discretization;
- K0 definition/version;
- wall/corner identification;
- longitudinal harmonic settings;
- thresholds;
- ODB;
- code commit;
- family-definition version.

Cached basis data may be reused only when its physical-definition hash matches exactly.

## 24. Acceptance criteria for implementation

The rewrite is complete only when all of the following are true:

1. New runs store U + UR and no automatic S/E/SF/SE.
2. Distinct physical pieces remain unconstrained across gaps in the reference model.
3. Bolt spacing is absent from K0/family-definition hashes.
4. Whole-section L/D/G bases are mechanically generated for all four pieces together.
5. Assembly is reported separately and is not subtracted before classification.
6. Seam opening/sliding/slip diagnostics are available.
7. Multi-harmonic modal content participates in classification.
8. K0-energy is the primary metric and vector/kinematic shares are secondary.
9. Repeated eigenvalues are handled as eigenspaces.
10. The current geometric classifier remains available only as an audit/screening comparison.
11. Legacy U-only ODBs fail mechanical classification explicitly rather than assuming zero rotations.
12. Existing build-only, full-run and resume-post command forms remain valid.
13. CAE, INP and downstream GMNIA workflow compatibility are preserved.
14. Synthetic, CUFSM/fcFSM, built-up and mesh-validation tests are present.
15. The output plot style continues to show actual piece-wise mode shapes comparable to the supplied `mode_sections.png`, with mechanical and diagnostic annotations separated.
