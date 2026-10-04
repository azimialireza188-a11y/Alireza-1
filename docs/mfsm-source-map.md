# Source map and unimplemented physics

This is a source map for the implemented **supplied-operator numerical kernel**,
not a declaration that all published mFSM subspaces have been reproduced.

| Implementation | Source / status |
|---|---|
| Compatible-space energy ratio and energetic rank reduction | Khezri–Rasmussen 2019 Part 1, equations 125/134; numerical construction implemented with supplied H |
| Total system metric includes elastic connectors; modal shell strain metric excludes connectors | SSRC 2023, section 4, equations 24/29/33–35; implemented through supplied system/components |
| Auxiliary nu=0 strain integration | 2019/2023 Poisson-coupling convention; implemented for explicitly supplied strain maps, not an Abaqus element reconstruction |
| Distortional transverse equilibrium, source modal ordering and warping orthogonality | NOT independently reproduced; H/constraints must be supplied; no automatic physical family builder |
| Closed-section modified torsion from Part 2 | Automatic derivation NOT implemented; supplied reviewed G flexural/torsional spaces now support numerical subtype QC |
| Exact rigid-link nullspace reduction | Project adaptation, not the paper's finite-fastener formulation; ideal small-rotation kernel implemented; actual BEAM_MPC equivalence pending |
| Base-state active contact / S4R shear, drilling and stabilization | NOT extracted; caller must supply verified compatible operators; primary activation blocked |
| Multi-term U/UR harmonic mapping | Explicit sin/cos fitting kernel only; automatic physical parity/interpolation NOT implemented; supplied inverse-map reconstruction QC is implemented |
| Invariant observed-cluster shares/bounds and signed cross-term QC | Project numerical reporting adaptation; tested with synthetic matrices, not a published physical benchmark |

The 2019 Part 1 author-uploaded text and the supplied 2023 article were inspected
for the energy-ratio kernel. The complete Part 1/Part 2 mechanical construction
was not independently transcribed or benchmarked; missing derivation coverage
is a blocker for automatic production activation, not permission to infer it
from eigenmode images.

The 2023 article's elastic Cartesian connections differ from this repository's
BEAM_MPC and general contact model. No fictitious spring stiffness is assigned
to MPCs and no contact-free tangent is silently substituted.

No AISI S100-2024 compliance or built-up DSM applicability claim follows from
these percentages. See the approved spec for the required physical validation.


Continuation source check: the author-uploaded 2019 Part 2 text was accessed
(DOI 10.1016/j.tws.2019.01.043), including its closed-cell modified torsion
criteria. Its downloadable full-text link failed, and mathematical text
extraction is not sufficient to independently transcribe all matrix operations.
The official 2025 SSRC applications paper was also inspected, section 4,
equations 26–27. It confirms that compatible H must enforce transverse
equilibrium and orthogonality; the energy-ratio kernel alone cannot generate
those restrictions. Neither source check is counted as completed physical
validation or an automatic hierarchy implementation.
