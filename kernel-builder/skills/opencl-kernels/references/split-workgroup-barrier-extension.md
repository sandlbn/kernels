# cl_intel_split_work_group_barrier — reference

Source: [Khronos registry v1.0.0](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_split_work_group_barrier.html) (status **Shipping**).

## Overview

Splits OpenCL C `barrier()` / `work_group_barrier()` into:

1. **Arrive** — work-item reached the barrier; execution continues.
2. **Wait** — block until all work-items in the **work-group** have arrived.

Motivation: overlap independent instructions with waiting (similar in spirit to C++20
split barriers / latches). **Sub-group split barriers are out of scope** for this
extension (resolved in spec issues).

## Host and device enablement

| Layer | Requirement |
|-------|-------------|
| Device | `cl_intel_split_work_group_barrier` in `CL_DEVICE_EXTENSIONS` |
| OpenCL C | `#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable` |
| SPIR-V (if loading SPIR-V) | Module declares `SPV_INTEL_split_barrier` and capability **SplitBarrierINTEL** |

No new OpenCL host API functions or enums.

## Builtins

```c
void intel_work_group_barrier_arrive(cl_mem_fence_flags flags);
void intel_work_group_barrier_wait(cl_mem_fence_flags flags);

void intel_work_group_barrier_arrive(cl_mem_fence_flags flags, memory_scope scope);
void intel_work_group_barrier_wait(cl_mem_fence_flags flags, memory_scope scope);
```

Overloads with `memory_scope` require OpenCL C 2.0 or newer.

### Memory ordering (informal, from spec issues)

- Arrive behaves like a **release** fence for the address spaces in `flags`.
- Wait behaves like an **acquire** fence; wait **synchronizes-with** other work-items' arrives in the same work-group.
- For **sequentially consistent** behavior, use a non-split barrier or insert explicit fences before arrive / after wait.

### `flags`

Bitfield, zero or:

| Flag | Address space |
|------|----------------|
| `CLK_LOCAL_MEM_FENCE` | `__local` |
| `CLK_GLOBAL_MEM_FENCE` | `__global` |
| `CLK_IMAGE_MEM_FENCE` | images; if used, `scope` must be `memory_scope_work_group` |

### `scope`

Describes which work-items participate in memory ordering. Default when omitted:
`memory_scope_work_group`.

If arrive and wait pass different `scope` values, constraints apply to work-items in the
**narrower** of the two scopes.

If arrive and wait pass different `flags`, visibility applies only to the **intersection**
of the two flag sets.

**For each call, `flags` and `scope` must be identical across all work-items in the
work-group.**

## Control-flow and pairing (undefined behavior if violated)

Same spirit as OpenCL work-group barriers:

1. If any work-item **arrives**, all work-items in the work-group must **arrive**.
2. If any work-item **waits**, all work-items in the work-group must **wait**.
3. If arrive is in a **conditional**, either no work-item enters it, or **all** do.
4. Same for wait in a **conditional**.
5. If arrive is in a **loop**, all work-items must arrive on the **same iteration**.
6. Same for wait in a **loop**.
7. A work-item must **not wait before it has arrived** (for that split-barrier instance).
8. After arrive, **no second arrive** before the matching wait.
9. After wait, **no second wait** before the next arrive.

Practical pattern: one arrive → optional independent code → one wait per logical barrier.

## SPIR-V environment (OpenCL SPIR-V)

When the OpenCL environment supports `cl_intel_split_work_group_barrier`, it must accept
SPIR-V modules using `SPV_INTEL_split_barrier` and **SplitBarrierINTEL**.

For `OpControlBarrierArriveINTEL` / `OpControlBarrierWaitINTEL`:

- Execution scope: **WorkGroup**
- Memory scope: same valid values as `OpControlBarrier`
- Arrive: memory semantics include **Release**
- Wait: memory semantics include **Acquire**

DPC++/SYCL toolchains often require enabling the SPIR-V extension at link time
(`-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier`) when kernels lower to these
instructions; pure OpenCL C compiled from source on Intel GPU typically handles this in
the driver stack when the device advertises the extension.

## Revision note

Spec version 1.0.0 (2026-01-06 build) adds the **intel_** prefix to the C built-in names
(earlier drafts used unprefixed names). Always use `intel_work_group_barrier_arrive` /
`intel_work_group_barrier_wait` in new kernels.
