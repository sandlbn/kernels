# Split work-group barriers (Intel OpenCL)

Use this skill when a kernel already needs a **work-group** barrier (`barrier()` /
`work_group_barrier()`) and you can run **memory-independent** instructions between
the arrive and wait phases to hide latency.

Do **not** use this extension for sub-group–only synchronization. The Khronos spec
explicitly does not define sub-group split barriers; these builtins synchronize **all
work-items in the work-group**, same as a normal work-group barrier.

## Estimate the ceiling first

A split barrier can only ever recover part of the time already lost *at* the
barrier, so bound the payoff before restructuring anything. Profile the kernel
and read the **barrier stall** percentage (VTune reports it directly under
"Stall Reasons"). That number is your absolute ceiling; the realistic gain is a
fraction of it.

Then ask which half of that stall you actually have:

| Cause | Symptom | Does splitting help? |
|-------|---------|----------------------|
| **Notification latency** — everyone reaches the barrier together and pays a fixed rendezvous cost | Barrier stall roughly constant per barrier, work per work-item is uniform | **Yes** — this is the case the extension targets |
| **Skew** — work-items genuinely arrive at different times | Barrier stall stays high even when you halve the barrier count; some work-items do more pre-barrier work (e.g. an elected loader) or wait on variable-latency global loads | **No** — the fast work-items still sit and wait for the slow ones |

Skew is the common case in staged GEMM-style kernels, and it is why a split
barrier can measure as exactly zero gain (see Profiling pitfalls). Fix skew by
balancing the pre-barrier work across sub-groups first; only then does splitting
buy anything.

## Quick workflow

1. Confirm `CL_DEVICE_EXTENSIONS` contains `cl_intel_split_work_group_barrier`.
2. No pragma is needed on Intel NEO — see [Enabling](#enabling) below.
3. Replace a single work-group barrier with `intel_work_group_barrier_arrive(flags)` → independent work → `intel_work_group_barrier_wait(flags)` when the middle section is safe.
4. Keep **identical** `flags` (and `scope`, if used) on the paired calls and on **every** work-item in the group.
5. Move **every read** of the data the barrier protects to *before* the arrive — not just out of the middle section. See [Recycling hazard](#the-recycling-hazard).
6. Measure: split barriers help only when the middle section is useful, long enough to overlap waiting, and the barrier stall is latency- rather than skew-dominated.

For launch sizing and SLM limits, see [architecture.md](architecture.md).

## Enabling

The builtins are available whenever the device advertises the extension.
Verified on NEO 25.35 by compiling `intel_work_group_barrier_arrive` /
`_wait` with **no pragma and no `#ifdef` guard** — it built and ran correctly.

```c
/* Optional, and only for portability to stacks that require it. On Intel NEO
 * this line emits "OpenCL extension ... unknown or does not require pragma -
 * ignoring" and is the only source of compiler-warning noise, so omitting it
 * gives a clean build. The macro is defined, so the #ifdef guard works. */
#ifdef cl_intel_split_work_group_barrier
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#endif
```

**`work_group_barrier` does not exist by default.** Intel's OpenCL C defaults to
version 1.2, where the OpenCL 2.0 spelling is undeclared and the compiler
unhelpfully suggests `sub_group_barrier` instead:

```
error: use of undeclared identifier 'work_group_barrier'; did you mean 'sub_group_barrier'?
```

Write `barrier(flags)` for the non-split fallback, or build with `-cl-std=CL2.0`.
The `intel_work_group_barrier_*` builtins themselves are unaffected and are
available under the 1.2 default.

## Concept

| Phase | Builtin | Role |
|-------|---------|------|
| Arrive | `intel_work_group_barrier_arrive` | Marks that this work-item reached the barrier; execution **continues**. Prior memory ops in the fenced address spaces are ordered like a **release** before other work-items observe them after the wait. |
| Wait | `intel_work_group_barrier_wait` | Blocks until **every** work-item in the work-group has arrived; then prior arrives **synchronize-with** this wait (**acquire**). |

With no independent work between arrive and wait, behavior matches a normal work-group
barrier for typical local/global fence combinations. The win is overlapping other
instructions with waiting for slow peers or memory.

## OpenCL C API

```c
// Pragma shown for spec completeness; not required on Intel NEO (see Enabling).
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

void intel_work_group_barrier_arrive(cl_mem_fence_flags flags);
void intel_work_group_barrier_wait(cl_mem_fence_flags flags);

// OpenCL C 2.0+ (needs -cl-std=CL2.0; the 1.2 default rejects memory_scope):
void intel_work_group_barrier_arrive(cl_mem_fence_flags flags, memory_scope scope);
void intel_work_group_barrier_wait(cl_mem_fence_flags flags, memory_scope scope);
```

`flags` is zero or a combination of `CLK_LOCAL_MEM_FENCE`, `CLK_GLOBAL_MEM_FENCE`, and
`CLK_IMAGE_MEM_FENCE` (image fence requires `memory_scope_work_group` when using `scope`).

If `scope` is omitted, it defaults to `memory_scope_work_group`.

There are **no** new host API enums or functions — only the device extension string and
kernel builtins.

## Mandatory usage rules

Treat these like standard work-group barriers (uniform control flow):

- If **any** work-item **arrives**, **all** work-items in the work-group must **arrive** (same for **wait**).
- Same rules for **conditionals** and **loops** as `barrier()`: every work-item must enter the same branch / loop iteration that hits the barrier.
- **Order:** arrive before wait; no second arrive before the matching wait; no second wait before the next arrive.
- **`flags` and `scope` must be the same for all work-items** in the work-group on each call.
- **Pairing:** use the same `flags` (and `scope`) on arrive and wait. If they differ, only the **intersection** of address spaces and the **narrower** scope apply (avoid this in new code).
- **Reads retire before the arrive:** no read of barrier-protected memory may sit between arrive and wait, even of a buffer this iteration is not writing. See below.

Full spec wording and SPIR-V notes: [split-workgroup-barrier-extension.md](split-workgroup-barrier-extension.md).

## The recycling hazard

This is the failure mode that a straight reading of the pairing rules does not
catch, and it produces wrong answers rather than a compile error.

`wait` blocks until every work-item has **arrived** — not until every work-item
has finished the code *after* its arrive. So the moment you clear your own wait,
a peer may still be executing its middle section. If your next action recycles a
buffer that the peer is reading in its middle section, you corrupt it.

It bites the most natural-looking multi-buffer pipeline:

```c
/* WRONG: reads of slm[cur] straddle the arrive. */
for (...) {
    read_half(slm[cur]);                            /* first half of the reads */
    write(slm[cur ^ 1], staged);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    read_half(slm[cur]);                            /* <-- still reading here  */
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    cur ^= 1;
}
```

On the next iteration this work-item writes `slm[cur ^ 1]`, which is the buffer
a slower peer may still be reading after its own arrive. Two buffers are not
enough to make that safe.

The fix is to hoist **all** reads of the protected buffer above the arrive, so
the middle section touches registers only:

```c
/* RIGHT: everything the barrier protects is already in registers. */
for (...) {
    operands = read_all(slm[cur]);                  /* all reads retire first  */
    write(slm[cur ^ 1], staged);
    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
    compute(operands);                              /* registers only          */
    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    cur ^= 1;
}
```

Now two buffers suffice, and the argument is short: a write to `slm[cur ^ 1]`
happens after this work-item's previous `wait`, which happens after every peer's
previous `arrive`, which happens after every peer's reads of that buffer.

Note the register cost — hoisting every read means the whole tile must be live
across the middle section at once. On a kernel that is already near its GRF
budget this can force a spill, which will cost far more than the barrier ever
did. Check for spilling before concluding the split barrier was a bad idea.

If you cannot hoist the reads, add a third buffer instead: with three, the
buffer being written was last read two iterations ago and two waits have since
intervened.

## Example: local tile publish with overlapped math

Each work-item stores its element to `__local` memory, arrives, runs work that does not
read the tile, waits, then reads a neighbor's slot (classic "fill SLM → sync → consume"
pattern with latency hiding in the middle).

```c
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable

__kernel void split_barrier_local_tile(
    __global const float *global_input,
    __global float *global_output,
    __local float *local_cache,
    int total_elements)
{
    const int local_id = get_local_id(0);
    const int global_id = get_global_id(0);
    const int local_size = get_local_size(0);

    if (global_id < total_elements) {
        local_cache[local_id] = global_input[global_id];
    }

    intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

    /* Must not read local_cache (or other work-items' local writes) here. */
    float independent = 0.0f;
    for (int i = 0; i < 10; i++) {
        independent += native_sin((float)global_id * (float)(i + 1));
    }

    intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

    const int neighbor_id = (local_id + 1) % local_size;
    float combined = local_cache[local_id] + local_cache[neighbor_id] + independent;

    if (global_id < total_elements) {
        global_output[global_id] = combined;
    }
}
```

## When split barriers are a poor fit

- Middle section **reads** data protected by the barrier (including `__local` written by peers).
- **Short** middle sections — overhead may exceed benefit; keep the plain barrier.
- **Skew-dominated** barrier stall — the fast work-items still wait for the slow ones, and splitting recovers nothing. Balance the pre-barrier work first.
- Barrier stall is already a **small share** of total stall — the ceiling is that share, so the restructuring rarely pays for itself.
- Need **sequentially consistent** fence behavior — use a full barrier or explicit fences around arrive/wait (see spec issues in [split-workgroup-barrier-extension.md](split-workgroup-barrier-extension.md)).
- **Sub-group** collective algorithms — use sub-group builtins / shuffles instead; this extension does not split at sub-group scope.

## Profiling pitfalls

- **Skew:** A correct split can still match a plain `barrier()` when stall is
  arrival skew (variable global-load latency, uneven pre-barrier work). Fix
  balance first; splitting does not shorten the wait for the slowest work-item.
- **Measure fairly:** A/B split vs plain in the **same** kernel. Gains from
  fewer barriers, double buffering, or staging are not from the extension.
- **No barrier:** If a safe design needs no work-group barrier, that variant
  is the ceiling — split barriers only help when you must keep one.
- **Pacing:** Barriers used only to cap drift (e.g. cooperative prefetch ahead
  of peers) help only when drift actually limits the loop. Profile whether stall
  is skew or memory latency before paying barrier cost; split is cheaper than
  plain, not free.
- **Other implementations:** Another library's split-barrier schedule reflects
  its own codegen and builtins; treat it as a hypothesis to profile, not proof
  for your OpenCL kernel.

## Related material

- Khronos extension: [cl_intel_split_work_group_barrier](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_split_work_group_barrier.html)
- SYCL / SPIR-V link flag `SPV_INTEL_split_barrier` (DPC++): covered in the xpu-kernels SYCL-TLA reference, not required for pure OpenCL C JIT from source.
