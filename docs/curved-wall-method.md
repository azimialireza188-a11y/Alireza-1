# Curved-wall screening method, version 2

This change corrects geometric segmentation and rigid-motion invariance. It
does not establish that the resulting L/D/G shares are mechanically validated.
No change to eigenvalues or modal shapes is involved in archive reclassification.

## Geometry and kinematics

Each source polyline is split at sharp turns of at least 30 degrees or compact
same-sign high-curvature bends. The default radius cutoff is 0.04 times the
piece perimeter. Smooth waviness and larger-radius curved lips remain within a
wall. Classification sensitivity uses fractions 0.03, 0.04 and 0.05. This is a
documented geometric assumption; stability across these three values does not
prove a universal wall definition. Detection never uses modal percentages.

Source wall endpoints map to existing mesh nodes. The report shows both and
records the maximum mapping distance. Unrepresented compact bends and tangential
residuals remain outside Local. Missing source geometry uses the previous
conservative mesh straight-run fallback.

For a wall point x, let t be its normalized arclength, c the endpoint chord,
and r = x - (x0 + t*c). The endpoint field is

    u_reference = (1-t)*u0 + t*u1 + omega*J*r
    omega = dot(J*c, u1-u0) / dot(c,c)

where J rotates a vector through 90 degrees. This exactly reproduces linearized
rigid translation and rotation of an initially curved wall. Local is the normal
part of `u - u_reference`. The coarse interpolator uses the same field.
Global and relative piece-rigid motion are separated using endpoint fits.
Distortional uses remaining coarse endpoint motion constrained against first-order
chord extension; this is not a full curved-shell inextensibility condition.
Other includes extension and the remaining unexplained displacement, so the
five components reconstruct the input.

Shares use squared displacement norms integrated over the saved section nodes
and longitudinal stations. L/D/G are normalized by their three self-norms;
Assembly and Other by all five self-norms. The components are nonorthogonal.
`component_norm_sum_over_input` and signed `displacement_cross_percent` expose
their overlap/cancellation. They are not strain-energy shares or portions of
critical load. The curvature index is a diagnostic of the corrected normal
residual, not shell curvature strain energy.

## Verification and limits

Tests cover smooth-wall continuity, compact/sharp bends, scale invariance,
zero Local under rigid motion, invariant Local under superposed rigid motion,
fixed-end normal bending, and symmetric channel flange rotation with no internal
wall bending. Sparse application is compared with the dense pipeline operators.
Archive order, provenance mismatch rejection and recomputed cross terms are
also tested. These checks verify the stated kinematics, not mechanical family
certification of the user's curved built-up section.

The archive command checks source geometry, model signature, ODB hash identity,
mode/eigenvalue tables and unchanged source file hashes. It re-evaluates complete
eigenvalue-cluster participation bounds and applies spectral isolation and
Assembly/Other gates. A single-mode Local share above 90% does not necessarily
produce a final Local label if its cluster is mixed or unresolved.

No independent mechanical reference, compatible strain-energy basis or mesh
convergence study was provided for this reclassification. DSM critical stresses
remain unaccepted. A mechanically validated reference for the same geometry,
loading and constraints is required before using these labels as certified
Local/Distortional/Global modes. Visual resemblance or a larger Local percentage
alone does not meet that requirement.
