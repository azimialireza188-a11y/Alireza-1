# Source map and remaining physical implementation

Complete user-provided 2019 PDFs are now available and their title/page counts
were verified: Part 1 (22 pages, journal 496–517, DOI 10.1016/j.tws.2019.01.041)
and Part 2 (14 pages, journal 518–531, DOI 10.1016/j.tws.2019.01.043).
Equation pages were rendered and visually checked, including Part 1 journal
507, 514, 515 and Appendix A1 journal 516; Part 2 journal 520–521.
The copyrighted PDFs and extracted full text are not republished in this repo.

| Source | SHA256 of inspected PDF |
|---|---|
| 2019 Part 1 | `25ad6acab08fc43d12244252f5724bd371111e29c3daeabf48b50f303c51fea8` |
| 2019 Part 2 | `4186ec878edf65c10818d30c2ae1184b6bd8ea2683ebcb7f1e66cc1fb579367b` |

| Implementation | Source / status |
|---|---|
| Compatible-space energy ratios | Part 1 125/134; SSRC 2023 35; existing supplied-H kernel retained |
| System metric includes supported connections; component criteria exclude them | SSRC 2023 24/29/33–35; supplied operator contract |
| Automatic GA and GB | Part 1 98–113; full coordinate space, eps_y warping orthogonality |
| Classical GT | Part 1 114–122; excludes GA/GB in eps_y metric |
| Modified closed-cell GT | Part 2 2–12; explicit reviewed strip split; zero open-strip shear only; spectral curvature selection adaptation |
| D | Part 1 123–131; warping orthogonality plus explicit transverse equilibrium; classical GT retained for D even when reported G uses modified GT |
| Canonical equilibrium row construction | Appendix A1: select K_kappa_x rows before multiplying reduction Q; canonical/reduced operator consistency checked |
| L | Part 1 132–137; eps_x/eps_y/gamma_xy energy nullspace |
| TES / TEP | Part 1 193–203; aggregate transverse shear adaptation for Eq200 |
| Aggregate S, closed cells | Part 2 Table1 CLOSED row: canonical Euclidean complement of other spaces, pulled back for changed/reduced coordinates |
| Aggregate S, open sections | Part1 139–146 plus 186–192, Table1 OPEN row: explicit warping choice Sw+SCt from reviewed canonical V projection, or supplied reviewed alternative aggregate; no closed-complement substitution |
| GA excluded from reported global buckling | Approved project adaptation; GA remains a separate internal extension share |
| Auxiliary nu=0 integration | Implemented for supplied strain maps; physical material unchanged |
| Exact rigid-link reduction | Project adaptation; ideal small-rotation kernel; actual BEAM_MPC equivalence pending |
| S4R shear/drilling/stabilization and active contact | Automatic physical extraction NOT implemented |
| Multi-term U/UR mapping | Supplied sin/cos kernel and inverse reconstruction QC; automatic physical parity/interpolation NOT implemented |
| Eigenspace invariance / signed cross QC | Numerical project adaptation, tested on synthetic matrices |

## Explicit mathematical adaptations

Part 2 Eq12 uses **K_kappa_y + K_kappa_xy**, not membrane strain/shear.
Selecting arbitrary candidate columns with positive diagonal curvature energy
is unstable under rotations mixing torsion and warping-only shear. The code
diagonalizes this generalized curvature ratio on the candidate span and keeps
its positive spectral subspace. This preserves the positive-curvature mechanism
while making the split invariant to candidate basis changes. It remains a
numerical interpretation requiring comparison with published closed-cell examples.

Part 1 Eq188–189 uses SDw while the surrounding description refers to
orthogonality against other transverse-only shear spaces. The open warping
selection uses the printed SDw directly: because gamma(D)=0 and D splits into
warping/transverse parts, gamma(SDw) equals minus gamma of its transverse part.
We do not change the printed equation or claim every secondary subdivision has
been individually reproduced. For Eq200 the code constructs the aggregate transverse
shear space: eps_x/eps_y-null vectors independent of L. This is intended to span
SBt+STt+SDt+SCt under the source strain/warping assumptions, making the secondary
shear ordering unnecessary. It excludes L explicitly before TEP is constructed.
Only aggregate S and TE are reported. Physical examples must verify the span
equivalence, especially after exact-MPC reduction. Overlaps are rejected by the
joint projector; they are never hidden by an arbitrary family reorthogonalization.

The open-section warping projection is reviewed input, not an inferred mask.
Idempotence, canonical metric self-adjointness and strain-map consistency are
checked. An exact-MPC reduction that does not preserve a compatible warping
projection cannot use this automatic open-S path; an explicitly reviewed source
aggregate is the supported alternative. Neither alternative is silently selected.

The complete coordinate space is used, never a truncated observed eigenbasis.
The hierarchy reuses one system Cholesky/whitening across its eigensystems;
all source criteria remain independent of connector contributions. Source and
coordinate/constraint provenance, equilibrium, topology, tolerance and algorithm
version bind the hierarchy cache and mechanically generated subtype identities.

## Remaining production transition

Automatic modal hierarchy on supplied operators is now implemented.
Automatic production ODB/INP-to-physical-operator generation is still missing.
The 2023 elastic Cartesian connections are not automatically equivalent to the
actual BEAM_MPC/contact model. No fictitious bolt stiffness or contact-free
tangent is substituted. Actual U+UR ODB/INP data and Abaqus tangent/equivalence
evidence are required for the remaining physical implementation/validation.

No published physical example, production Abaqus model or GPU runtime has been
validated on this host. Numerical tests do not establish AISI S100-2024
compliance or built-up DSM applicability. Primary/default activation stays
disabled as required by the approved specification.
