# Curved-wall reclassification implementation plan

**Goal:** Correct physical-wall recognition and rigid-motion invariance, then
reclassify the existing A3984 archive without building a model or running Abaqus.
**Spec:** The four-step design accepted in this conversation: identify walls,
validate curved-wall kinematics, compare modes 2/124/197, process all saved modes.
**Architecture:** A source-geometry wall detector feeds the existing projector.
An archive-only command reads saved NPZ/JSON/CSV and writes a separate report.
**Tech stack:** Python, NumPy, SciPy sparse, Matplotlib; no Abaqus dependency in
the archive command.

## Constraints and review focus
- Preserve all original CAE/INP/ODB/results and existing user edits/deletions.
- Work in the requested git_hub checkout; no commits, resets, or solver calls.
- Physical segmentation must not depend on eigenmode values or desired labels.
- Smooth waviness and curved lips must survive; compact bends remain separate.
- Infinitesimal rigid motion of a curved wall must have zero Local component.
- Missing source geometry retains the conservative existing fallback.
- Keep sensitivity, eigenspace ambiguity and non-mechanical status explicit.

## Tasks
- [x] Add failing tests for smooth walls, compact/sharp bends, scale invariance,
      curved rigid motion, fixed-end local bending and coarse wall motion.
- [x] Implement source-wall detection and rigid-exact wall interpolation;
      use the same interpolation for Local removal and coarse reconstruction.
- [x] Add an archive-only reprocessor with provenance checks, synthetic controls,
      wall overlay, selected-mode comparisons, all-mode CSV/JSON/HTML, and
      updated eigenspace/sensitivity information. Never overwrite source reports.
- [x] Run tests, selected-mode review, all 250 saved modes, inspect figures,
      obtain independent code review and record outcomes.

## Ledger
- Initial evidence: 28 short walls, 48.4% coverage, 64 fold nodes; max L=3.74%.
- Ruling: preserve the previous in-place changes and user-deleted code-change
  files, because this task continues work in the explicitly requested checkout.
- Ruling: postprocessing is authorized; full Abaqus/model execution is forbidden.
- Outcome: 110 tests passed. Both selected 3-mode and full 250-mode archive
  commands completed without Abaqus. Independent review found plotting/schema
  issues and a stale cross-term diagnostic; all were corrected and exercised.
- Geometry: 16 walls, 96.4454% coverage, maximum endpoint mapping 3.8268 mm.
  Radius fractions 0.03/0.04/0.05 give identical classifications on this dataset.
- Full report: modal_reclassification_v2 under the original A3984 run.
  Eigenvalues, stresses and wavelengths match the original 250-mode audit;
  source hashes, interactive payloads and local report links verified.
- Individual labels: 16 L, 7 D, 206 Mixed, 21 Assembly. Conservative final
  labels: 0 L, 3 D, 130 Mixed, 8 Assembly, 109 Unresolved. Mode 197 has 95.9511%
  individual Local but its 8-mode cluster ranges from 12.5385 to 95.9516% Local;
  its final label remains Unresolved. No claim of Local-family absence follows.
- Scientific boundary: analytical kinematic controls pass; no independent
  mechanical reference or mesh-convergence certification was supplied or run.
  All mechanical eligibility flags remain false and DSM inputs unaccepted.
