# Full-resource execution

Defaults are `--cpus auto --gpus auto`. CPU auto uses all visible logical CPUs;
GPU auto discovers visible NVIDIA devices. There is no fixed CPU ceiling, 24 GB
memory limit, RAM reserve percentage or GPU reserve. Explicit numeric overrides
remain available for repeatable comparisons.

The Abaqus job requests memory **100 PERCENTAGE**,
`getMemoryFromAnalysis=False`, and all selected CPUs. `numGPUs` is passed only
when the installed Job API advertises that keyword; support and the request are
recorded. A GPU count request does not prove that Abaqus's selected buckling
solver uses GPU acceleration. CPU-only Abaqus builds continue to work.

NumPy/SciPy BLAS threads use the resolved CPU count. The standalone CPU work queue
limits nested BLAS to one thread per concurrent task; the mFSM audit instead
uses all-core BLAS on batches. Optional CuPy FP64 projection is timed on every
usable device including transfer and synchronization, and compared numerically
with CPU results. Faster GPUs receive dynamically scheduled batches, one active
batch per device. ODB extraction stays on the parent thread.

Install compatible NumPy, SciPy, threadpoolctl and matplotlib in the runtime that
runs the audit. CuPy matching the installed CUDA runtime is optional; absent or
unsupported GPU runtime yields explicit CPU fallback. The original screening
path does not require CuPy. No GPU hardware was available for this implementation's
verification run.

Batch sizes use live available RAM and estimated concurrently live working sets,
with zero arbitrary safety reserve. Reduced mode history is disk-backed.
Allocation failures during projection retry one mode at a time on CPU using
already-read U/UR. A system unable to hold even one mode or the dense operators
fails explicitly; no mathematical tolerances or model physics are weakened.
Full utilization depends on the algorithm and installed solver capabilities;
allocating unused memory or doing slower GPU work is not an optimization.

JSON records inventory, selected resources, zero reserve, backend timing/fallback,
cache identity, batch size, retries and elapsed time. Use actual production timing
to assess speed; the repository does not claim an unmeasured optimal runtime.
