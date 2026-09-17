# Fast Model Switching with a Shared Host-Memory Weight Cache

Loading a model is usually treated as one operation, but it combines several
different costs: starting the serving process, constructing the model,
reading its checkpoint, moving weights to the GPU, and preparing runtime state.
Only some of these costs need to occur during a model switch.

I built a prototype that keeps the serving process and model structures warm,
caches one copy of each model's weights in shared host memory, and transfers
only the selected weights to the GPU. On a 40 GB NVIDIA A100, the copy from
pinned host memory to the GPU accounted for more than 90% of a switch at the
largest weight set I tested. Attaching those tensors to a prebuilt model cost
about 1% of the copy time with Hugging Face Transformers and about 3% in my
vLLM prototype.

These are weight-switch measurements, not complete cold-start or
time-to-first-token measurements. They exclude scheduling, process startup,
tokenizer loading, compilation, CUDA graph capture, KV-cache initialization,
distributed rendezvous, and warmup.

## Keep the runtime warm and switch only the weights

A conventional cold start follows a long path:

```text
start process -> import framework -> construct model -> read checkpoint
              -> copy weights to GPU -> initialize runtime -> serve
```

My prototype moves the reusable work out of the switch path:

```text
once per process:  construct model skeletons and initialize the runtime
once per node:     cache model weights in shared host memory
on each switch:    release old GPU weights -> copy new weights -> attach them
```

This changes the problem from "start another server" to "move one prepared set
of bytes into GPU memory." It is most useful when a worker must alternate among
a known set of full models and only one model needs to occupy GPU memory at a
time.

## Build one shared, pinned host cache

The host cache combines three mechanisms:

1. A loader writes each checkpoint once to a file in a memory-backed `tmpfs`,
   such as `/dev/shm`.
2. Each serving process maps that file with `mmap`. The processes have
   different virtual addresses, but the mappings refer to the same underlying
   physical pages, so the node does not need a private host copy per process.
3. Each serving process registers its own mapped address range with
   [`cudaHostRegister`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).
   Registration page-locks the backing memory so CUDA can transfer it to the
   GPU without first copying it through a temporary pinned staging buffer.

![Two serving processes on one node map and register their own views of the same cached checkpoints. The host pages are shared; each activation still performs one host-to-device copy into that process's GPU. Of the three phases in a switch, only the copy scales with weight-set size.](assets/model-switch-shared-host-cache.svg)

The term *zero-copy* needs care here. `mmap` avoids duplicate host copies
across processes. Loading weights onto the GPU still requires a host-to-device
copy over PCIe; the benefit of pinning is that this copy can use DMA directly
from the shared host pages.

Registration and page faulting can themselves be expensive, so the prototype
performs both before the timed switch. Before registration, pages in a `tmpfs`
mapping are not locked and may be swapped; a successful `cudaHostRegister`
call locks the range for the lifetime of the registration. The call fails if
the locked-memory allowance is insufficient, and because each process
registers its own virtual mapping, each requires an allowance for the full
range even though the physical pages behind them are shared.

## Prebuild model structures on the `meta` device

Fast access to weight bytes is not enough if every switch reconstructs the
model. PyTorch's [`meta` device](https://docs.pytorch.org/docs/stable/meta.html)
allows me to instantiate modules whose parameters contain shape and dtype metadata
but no storage. I create these model skeletons when the serving process
starts.

During activation, the prototype copies the target tensors from the pinned
host cache to GPU storage, then calls
[`load_state_dict(assign=True)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict).
With `assign=True`, the module adopts the state-dictionary tensors instead of
copying them into pre-existing parameter storage. That matters because a meta
parameter has no storage into which PyTorch could copy.

The control flow is roughly:

```python
# Illustrative pseudocode; cache layout and release logic are model-specific.
with torch.device("meta"):
    model = ModelClass(config)

def activate(model, mapped_host_state):
    gpu_state = copy_cached_tensors_to_gpu(mapped_host_state)
    model.load_state_dict(gpu_state, strict=True, assign=True)
    model.tie_weights()  # if the architecture uses tied parameters

def deactivate(model):
    replace_parameters_and_buffers_with_meta_tensors(model)
```

The serving layer must stop new work and wait for outstanding GPU operations
before replacing live weights. It must also preserve tied parameters, buffers,
quantization metadata, aliases, and any model-specific runtime state. The
release function above is deliberately pseudocode; it is not a generic
Transformers or vLLM API.

## Measured switch phases

I evaluated three weight sets on one 40 GB A100, spanning a 14× range in
size. Below, they are referred to by size relative to the smallest. The point of
interest is how a switch scales with weight-set size rather than how fast one
particular model loads, so the sets are identified only by that ratio.

The cache was already mapped, faulted in, and registered before measurement,
and each model structure had already been created. For each model, I measured:

- pinned host memory to GPU transfer;
- `load_state_dict(assign=True)`; and
- replacement of the active parameters with meta placeholders, making their
  GPU storage reclaimable by PyTorch's allocator.

I also ran inference and checked the output after each activation, but that
validation time is excluded from the figures below.

The copy is the only phase that scales with the weight set. It uses
host-to-device DMA from pinned pages, so, to first order,

```text
T_switch = T_copy + T_attach + T_release,      T_copy = S / B
```

where `S` is the size of the selected weights and `B` is the achievable
host-to-device bandwidth of the link. Across that range of `S`, my implied `B`
varied by less than 3%. That is what a saturated link looks like: per-call
overhead from copying the state dictionary one tensor at a time would have
penalized the smallest set the most, since it has the fewest bytes per copy,
yet the observed rate was flat. `B` was close to the practical throughput of
this host's PCIe link, so the copies were likely already near its limit
rather than leaving easy headroom on the table.

The other two phases are framework bookkeeping, and both are essentially
independent of `S`: attaching is a pointer assignment through
`load_state_dict(assign=True)`, and releasing swaps parameters for meta
placeholders. Their absolute cost barely moved across the full range of weight
sizes, so as a share of the copy they shrink rapidly:

```text
                      smallest set     largest set
T_attach  / T_copy      11 - 13 %        1 - 3 %
T_release / T_copy      10 - 14 %        2 - 5 %
combined                23 - 25 %        3 - 8 %
```

Each range spans the two runtimes: for the largest size, Transformers sits at
the lower end of the combined figure and the vLLM prototype at the higher end.
Equivalently, for the largest weight set, the copy alone accounted for 96.8% of
the measured switch under Transformers and 92.5% under the vLLM prototype.

Because `T_attach` and `T_release` stay roughly constant while `T_copy` grows
with `S`, their share falls like `1/S`:

```text
(T_attach + T_release) / T_switch  ->  0      as S grows
T_switch                           ->  S / B
```

So for any model large enough that switching it is worth engineering, the
switch is a bandwidth problem. Optimizing the framework attachment layer has a
bounded payoff, and that bound is already small.

I did not retain the host topology, so I cannot confirm the link generation.
A newer-generation link would perform the same copies proportionally faster, which
would also raise the share taken by attachment and release. If the link is
already saturated, the remaining levers are batched copies, multiple copy
streams, and NUMA-aware placement. Anyone reproducing this should record the
PCIe generation first, because it decides whether there is anything left to
optimize.

## Related work

vLLM provides a maintained
[`sleep`/`wake_up` lifecycle](https://docs.vllm.ai/en/latest/features/sleep_mode/)
for releasing and restoring a model's GPU memory without restarting the server,
and the project describes its use for model switching in
[Zero-Reload Model Switching with vLLM Sleep Mode](https://vllm.ai/blog/2025-10-26-sleep-mode).
If that lifecycle covers a deployment's needs, it is the supported path, because
vLLM owns the allocator, KV cache, distributed workers, and model-specific
runtime state.

This prototype began as a side project in August 2025.

## Practical boundaries

This design shifts cost rather than eliminating it:

- Host RAM must hold the cached catalog, and pinned pages reduce memory
  available to the rest of the node.
- Concurrent switches share PCIe and memory bandwidth, so independently fast
  transfers may slow down when they overlap.
- The cache is node-local. Sharing weights across nodes requires a different
  storage or networking layer.
- I measured single-GPU switches only. Under tensor parallelism each rank
  needs its own shard, so a deployment must decide whether ranks map the same
  cache file and register only the bytes they own, or whether the cache stores
  pre-sharded copies. Sharding on the switch path would put that work back into
  the critical section the design is trying to empty.
- A full serving transition must also reset or rebuild KV cache, CUDA graphs,
  compiled kernels, tokenizer/configuration state, and scheduler state as
  required by the engine.
- Replacing parameters with meta tensors returns their allocations to
  PyTorch's caching allocator, not necessarily to the CUDA driver. Repeatedly
  switching differently sized models can fragment the allocator; the short
  release measurements above did not test long-running fragmentation.
- Access permissions and cache invalidation matter because every worker reads
  the same physical backing pages.

## Takeaway

Fast model switching becomes simpler when weight placement is separated from
process startup and model construction. A shared `tmpfs` mapping avoids
duplicating the host cache, CUDA registration enables direct DMA from pinned
pages, and meta-device skeletons plus `assign=True` keep framework bookkeeping
off the critical path.

In this prototype, the largest switch was dominated by the host-to-GPU
transfer; attaching and releasing the weights together cost a single-digit
percentage of it. That does not make every model server instantly ready, but it
identifies the dominant measured operation and a practical architecture for
keeping the rest of the work out of the switch path. Whether that transfer can
be made faster depends mostly on the host link: throughput
held constant across weight sizes, which is what a bandwidth-limited copy looks
like.
