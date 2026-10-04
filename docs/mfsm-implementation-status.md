# SDD ledger — plan: docs/superpowers/plans/2026-10-04-builtup-mfsm-aggressive-execution.md
Pre-flight: tasks 1→3→4→6→7 share OperatorPack; task 5 maps DOFs; task 8 consumes ClassificationResult. Use immutable pack metadata and consistent FP64 dimensions.
Ruling: use unittest instead of pytest — repository uses unittest and this host has no pytest — equivalent discovery/tests, no scientific cost.
Ruling: clone on isolated feature branch is the workspace — direct GitHub integration is explicitly authorized — no modification of user local checkout.
Baseline: python -m unittest discover -q: 117 tests pass.
Source: author-uploaded 2019 Part 1 is readable via web, PDF link unavailable; equations 125/134 and 2023 Eq35 confirm core energy-ratio construction. Do not claim full topology/closed-torsion derivation independently reproduced.
Ruling: current-model automatic activation must remain unavailable without reconstructed active contact/MPC operators and physical benchmarks — mandated by approved spec — current production classifier remains screening until that evidence exists.
Task 1: contracts implemented; full source/physical benchmark evidence pending. 3 contract tests pass.
Task 2: implemented runtime discovery, zero-reserve batch capacity, all-CPU defaults, Abaqus memory100/getMemoryFromAnalysisFalse and U/UR. Tests 6 resource +23 pipeline pass. Actual Abaqus Job execution unavailable here.
Ruling: serializable resource provenance is stored in CLI Namespace — existing pipeline writes vars(args) to JSON — avoids nonserializable dataclass state.
Tasks 3–8: implemented supplied-map reference/strain integration, exact ideal rigid-link reduction, explicit harmonic mapping, source-supplied search-space energy-ratio basis, content-addressed atomic cache, energy projection/eigenspace bounds, CPU/GPU scheduling, independent assembly diagnostics and separate ODB audit reports.
Ruling: source-specific S4R/search hierarchy, Abaqus MPC/contact reconstruction and global subtype are not guessed — source fidelity and physical equivalence cannot be established on this host — numerical backend requires supplied reviewed operators; production primary activation remains pending.
Ruling: use one all-core BLAS CPU stream for the audit and one task per measured-faster GPU — avoids nested all-core CPU oversubscription and slower device work — actual best topology remains production-benchmark dependent.
Task 9: synthetic numerical timing measured (300 DOFs/100 modes, 9 CPUs, no GPU), closure 1.42e-15; physical validation manifest/commands remain PENDING. Assertions alone cannot activate primary method.
Task 10: CLI migration including resume, full resource/scientific limitation docs; root unittest discovery now includes new tests. 158 tests pass. Abaqus/GPU actual hardware checks unavailable.
Final review: fresh-context gpt-6-astra reviewer found 5 Important defects, no Critical. All 5 reproduced RED before fix.
Final: fixed amplitude-sensitive search/family/cluster rank — equivalent-scaled search and family/eigenspace tests RED→GREEN; normalized columns before energetic/SVD rank decisions.
Final: fixed batch-amplitude-dependent constraint masking — per-mode constraint test RED→GREEN; both raw and reduced mapping checks now per-column.
Final: fixed cluster signed-cross-term QC bypass — singleton counterexample RED→GREEN; rotation-invariant symmetric cross-operator bounds gate stable labels.
Final: fixed ignored mFSM CLI thresholds — ODB fixture custom dominance/cluster test RED→GREEN; forwarded and recorded all QC values.
Final: fixed screening SciPy dependency regression — import-blocked subprocess test RED→GREEN; numerical imports lazy at evaluate.
Final verification: python -m unittest discover -q → 163/163 pass; git -c core.whitespace=cr-at-eol diff --check clean.
Final: minor (deferred): sidecar report writes are not atomic; existing fresh output-directory contract retained.
Final: Ruling: absent automatic hierarchy/S4R/MPC/contact, inverse harmonic mapping, global subtype and integrated assembly benchmarks remain implementation gaps — do not present them as merely unavailable hardware evidence — backend remains explicit numerical prototype; cost is unfinished full production transition.
Final: minor (deferred): live VRAM sizing, measured simultaneous working sets/peak memory, topology tuning and representative full-batch GPU timing remain incomplete; actual GPU execution is unverified.
