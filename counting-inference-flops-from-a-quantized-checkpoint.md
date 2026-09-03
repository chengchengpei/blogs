# [WIP] How to Count Inference FLOPs from a Quantized Checkpoint

*A learning note in three parts: first the method, then the method applied to
a classical Transformer whose answer is known, then applied for real to a
quantized sparse-MoE model (DeepSeek-V4-Flash) — down to the FLOPs of one
complete request and an MFU number. Corrections welcome.*

## The method

Everything rests on one rule. A matrix multiply against a learned weight
`[out, in]` costs one multiply and one add per weight element used:

```text
FLOPs/token = 2 × logical GEMM parameters executed for that token
```

Why start from the checkpoint rather than write a formula from `config.json`?
Because modern configs stop mapping onto textbook formulas exactly when it
matters: DeepSeek-V4-Flash has no `kv_lora_rank`, no `qk_nope_head_dim`, none
of the fields a DeepSeek-V3-style formula needs — but every stored weight's
name, shape, and dtype sit in the checkpoint headers.

The recipe, in six steps:

**Step 1 — read the weight inventory.** A safetensors file starts with a
little-endian `uint64` header length followed by a JSON header listing every
tensor's name, dtype, and shape. Read those and stop — no tensor data, no
framework, no GPU; the cost scales with header metadata, not with the
gigabytes of weights. Keep all three fields per tensor. All three matter.

**Step 2 — translate stored shapes into logical GEMM shapes.** Quantized
checkpoints do not always store one element per logical weight: sub-byte
formats pack several values per stored element, and some formats pad. The
dtype and the quantization config tell you the layout; the inference engine's
loader code is the ground truth for how to undo it. Dtype and packing are part
of the operation schema — shape alone is insufficient.

**Step 3 — weight each matrix by execution frequency.** Stored is not
executed: a MoE layer stores all experts but runs `top_k` of them per token;
an embedding is a lookup (zero GEMM FLOPs); an LM head runs only where logits
are produced; auxiliary modules (e.g. multi-token prediction) count only if
that path is enabled. Summing over every weight matrix $W$, with $m(W)$ as its
execution multiplicity (how many times it runs per token):

$$
G_{\text{linear}} \;=\; \sum_{W} 2 \cdot \mathrm{numel}_{\text{logical}}(W) \cdot m(W)
$$

Notation: $G_{\ast}$ names are per-token FLOP totals ($G$ for GFLOP-scale
"GEMM work"). $G_{\text{linear}}$ is the model's learned-matmul cost for one
token; $G_{\text{lm}}$, used below, is the LM head's cost per application.


**Step 4 — add the matmuls that have no learned weight.** Parameter counting
cannot see matmuls between two *activations*: attention scores (`QKᵀ`),
attention aggregation (`softmax(QKᵀ)V`), and any sparse-selection scoring.

The costing rule is the same one — a matmul of $[m,k]\times[k,n]$ costs
$2mkn$ regardless of whether one side is a stored weight. What changes is
where the dimensions come from: a weight's dimensions sit in the checkpoint
header, while an activation-matmul's dimensions are *runtime* quantities read
from config and position. Concretely, one attention query with $N$ heads of
width $H$ attending to $k$ keys costs $2NHk$ for $QK^{\top}$ (per head: a
$[1,H]\times[H,k]$ matmul) and another $2NHk$ for $\mathrm{softmax}(QK^{\top})V$.
So the procedure is: identify each activation-by-activation matmul from the
architecture, write $2 \times (\text{product of its three dimensions})$, and
take the dimensions from config plus the query's position in the sequence —
which is why these terms depend on the *executed* attention schedule
(Application 2, Step 4 does this for a sparse model).

**Step 5 — assemble a request and compute MFU.** For $I$ prompt tokens and
$O$ generated tokens, sum over real positions:

$$
\text{prefill} \;=\; I \cdot G_{\text{linear}} \;+\; \sum_{p=0}^{I-1} \mathrm{attn}(p)
$$

$$
\text{decode} \;=\; O \cdot G_{\text{linear}} \;+\; \sum_{t=0}^{O-1} \mathrm{attn}(I{+}t) \;+\; O \cdot G_{\text{lm}}
$$

$$
\mathrm{MFU} \;=\; \frac{\text{total FLOPs}}{\text{elapsed seconds} \;\times\; N_{\text{GPU}} \times \text{peak FLOPs/s per GPU}}
$$

Here $\mathrm{attn}(p)$ is the Step-4 weightless-matmul cost for one query at
position $p$, summed over all layers. The sums matter because attention cost
grows with position — one "average token" shortcut would miss the causal ramp.

**Step 6 — validate against something your parser can't share.** Agreement
between two implementations is not enough if both can inherit the same wrong
assumption. Effective checks are derived independently of the parsing path: a
closed-form identity from pure config dimensions, an expected tensor census,
fail-closed handling of unknown layouts — and a known-answer model, which is
where we start.

## Application 1: a classical Transformer (known answer)

In [JAX Scaling Book](https://jax-ml.github.io/scaling-book/transformers/)
notation — batch `B`, query length `T`, key length `S`, model width `D`, FFN
width `F`, query heads `N`, KV heads `K`, head width `H`, vocabulary `V`,
layers `L` — a gated Transformer stores `3DF` MLP weights per layer (gate, up,
and down projections) and `2D(N+K)H` attention-projection weights per layer
(Q and O over `N` heads, K and V over `K` heads). That is Steps 1–3 with the
trivial case of the rule: no packing, and every weight runs once per token, so
each stored element costs exactly 2 FLOPs per token — $3DF$ weights become
$6BTDF$, and so on. Step 4 adds the weightless attention term $4BTSNH$
($2BTSNH$ for $QK^{\top}$ plus $2BTSNH$ for $\mathrm{softmax}(QK^{\top})V$).
Together:

$$
C_{\text{forward}} \;=\; L\left(6BTDF + 4BTD(N{+}K)H + 4BTSNH\right) + 2BTDV
$$

$C_{\text{forward}}$ is the total forward-pass matmul compute for the whole
batch, in FLOPs — $C$ for "compute cost," the scaling-law convention (as in
$C \approx 6ND$). It differs from the $G_{\ast}$ names above only in scope:
$C$ covers a whole batch of $B \times T$ tokens, while $G_{\ast}$ are
per-token.

For the book's example (`D=4096`, `F=16384`, `V=32000`, `L=64`, MHA with
`NH=D`, `B=1`, `T=S=2048`) this yields **75.304 TFLOP** forward — exactly one
third of the book's **225.911 TFLOP** training count, matching its
forward-plus-two-backward convention. (A causal kernel that skips the upper
triangle lowers the forward total to 73.105 TFLOP.)

The rule reproduces the published formulas. Now the model it was actually
built for.

## Application 2: DeepSeek-V4-Flash

### Steps 1–2: the inventory, and a packing trap

The model's `config.json` declares `expert_dtype="fp4"` (non-expert components
are FP8), and the routed-expert tensors carry dtype `I8` — the safetensors tag
for one-byte integer storage. Each stored byte packs **two** 4-bit FP4 values
along the GEMM K dimension (the input/reduction axis). Each routed expert is a
gated MLP, so it stores three matrices — the same `3DF`-per-layer pattern as
Application 1, just per expert:

| Weight (per expert) | Stored | Logical GEMM |
|---|---:|---:|
| `w1` (gate) | `I8 [2048, 2048]` | `[2048, 4096]` |
| `w2` (down) | `I8 [4096, 1024]` | `[4096, 2048]` |
| `w3` (up) | `I8 [2048, 2048]` | `[2048, 4096]` |

The pinned
[vLLM MXFP4 loader](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py#L73-L76)
confirms the layout: it doubles the stored dimension with the comment
`* 2  # weight is FP4-packed`. (My first pass skipped Step 2, summed physical
shapes, and under-counted the model by about 25% — the Step 6 checks below
catch this class of error.)

### Step 3: the linear term

Execution multiplicities here: six of the 256 routed experts per token
(`num_experts_per_tok=6`); shared expert and router once; embedding as a
lookup; LM head per generated token; MTP weights excluded (path disabled).
For 43 layers, hidden 4096, MoE intermediate 2048:

```text
routed experts   43 × 6 × 3 matrices × 2 × 4096 × 2048 = 12.986 GFLOP/token
everything else  attention projections, shared experts,
                 routers, head-compression projections  = 12.499 GFLOP/token
                                              G_linear  = 25.485 GFLOP/token
```

A satisfying cross-check: `25.485 / 2 ≈ 12.7B` GEMM-active parameters,
consistent with the model card's "13B activated" once you remember the
embedding and MTP are in the card's count but not in per-token GEMM work. The
`2 × 13B = 26 GFLOP/token` shortcut lands only ~2% high for this model — but
that is an outcome you verify, not a rule you assume.

### Step 4: the weightless matmuls

Per layer, for one query attending to $k$ keys while the sparse indexer scores
$c$ candidates:

$$
\begin{aligned}
QK^{\top} &= N \cdot 2 \cdot H_{qk} \cdot k \\
\mathrm{softmax}(QK^{\top})\,V &= N \cdot 2 \cdot H_{v} \cdot k \\
\text{indexer } q \cdot K &= N_i \cdot 2 \cdot H_i \cdot c
\end{aligned}
$$

($H_{qk}$/$H_v$ are the query-key and value head widths — both 512 here;
$N_i$/$H_i$ are the sparse indexer's head count and width.)

This model mixes local-window, selected-compressed, and pooled-compressed
attention layers. A uniform approximation (every query capped at
`index_topk + sliding_window` attended keys) is simple and stable for
comparisons; a faithful per-layer schedule yields a smaller total, because the
uniform cap over-counts the layers that attend to fewer keys. Elementwise
work (normalization, softmax, activations, routing, top-k) is disclosed as
omitted — each is a sub-percent rider on the matmul it accompanies.

The LM head adds **1.059 GFLOP** per application
(`2 × hidden × vocab = 2 × 4096 × 129,280`). Charging it once per *generated*
token is a convention: the first token's logits physically follow the last
prompt position, so the convention moves one application between phases
without changing request-level work.

### Step 5: one full request, then MFU

Worked example, `I=1000` prompt tokens, `O=200` generated tokens (attention
under the uniform approximation):

```text
prefill_linear   1000 × 25.485 GFLOP         = 25.485 TFLOP
prefill_attn     Σ over 1000 causal positions =  2.807 TFLOP
decode_linear     200 × 25.485 GFLOP         =  5.097 TFLOP
decode_attn      Σ over 200 decode positions  =  0.876 TFLOP
decode_lm_head    200 × 1.059 GFLOP          =  0.212 TFLOP
total                                         = 34.477 TFLOP
```

Illustration (hypothetical throughput): if a server completed 1,000 such
requests in 100 seconds on eight H100s against the dense-FP8 peak
(1,978.9 TFLOP/s per GPU), MFU = `1000 × 34.477e12 / 100 / (8 × 1.9789e15)`
≈ **2.2%** — decode-heavy serving is memory-bound, so low single digits are
normal. The denominator is a *reporting choice*: this model mixes FP4 expert
storage with FP8 components, so quoting the FP8 peak is a stated convention,
not a claim that every kernel has one precision. Report throughput and latency
alongside MFU.

### Step 6: the independent checks

The obvious validation is to write the calculator twice, independently, and
require the two to agree. I did that — and it failed in an instructive way:
both implementations agreed to relative tolerance `1e-9` while making the
*same* packed-shape mistake, because both read the same stored shapes and
trusted them. Agreement only proves the implementations match; it cannot catch
an assumption they share.

So the useful checks are ones whose answer does **not** come from parsing the
checkpoint at all:

1. **A closed-form identity from config alone.** Routed-expert work must equal
   `layers × top_k × 2 × 3 × hidden_size × moe_intermediate_size`. Every
   factor comes from `config.json`, none from a stored shape — so a parser
   that mis-reads shapes disagrees with it immediately. (This single identity
   would have caught the 25% under-count.)
2. **A tensor census.** Exactly `layers × n_routed_experts × 3 = 33,024`
   packed expert matrices must be found and decoded, each to the logical
   dimensions the config implies. Missing or extra tensors surface as a count
   mismatch instead of a silently different total.
3. **Fail closed.** An unrecognized packed layout raises an error rather than
   being guessed at — a wrong guess here produces a plausible number, which is
   worse than a crash.
4. **Invariants on what must *not* change.** Shared FP8 experts keep their
   stored shapes (they are not packed), and MTP weights stay excluded.
5. **A known-answer model.** Application 1 pins the costing rule itself to a
   published reference, independent of this model entirely.

Against the pinned checkpoint, the corrected implementations decoded all
33,024 packed matrices, reproduced 12.9856 GFLOP/token of routed-expert work,
and matched every request field at relative tolerance `1e-9`.

## Related approaches

Formula tools such as the JAX Scaling Book and
[llm-analysis](https://github.com/cli99/llm-analysis) start from architecture
constants; [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)
uses a compact parameter heuristic. Runtime tools such as
[DeepSpeed FLOPs Profiler](https://deepspeed.readthedocs.io/en/latest/flops-profiler.html),
[fvcore](https://detectron2.readthedocs.io/en/stable/modules/fvcore.html#fvcore.nn.FlopCountAnalysis),
and [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) inspect
an executing or traced model, but custom kernels can escape operator coverage.
[calflops](https://github.com/MrYxJ/calculate-flops.pytorch) can build models on
a meta device without allocating real weights.

I did not find an off-the-shelf tool combining header-only metadata,
quantization-aware shapes, sparse-MoE multiplicities, and explicit activation
matmuls. This method is lightweight, not architecture-free.

## Takeaways

1. Count `2 × logical GEMM parameters executed`, not stored elements — dtype
   and packing are part of the schema.
2. Execution frequency comes from config and model code: `top_k` experts,
   lookup embeddings, decode-only LM head, disabled MTP.
3. Add attention-score and indexer matmuls from the executed sparsity
   schedule; they are invisible to parameter counting.
4. Validate with a formula your parser cannot share, and state the MFU
   denominator convention explicitly.

## Reproducibility

- [Checkpoint revision `60d8d707`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)
- [vLLM commit `568afb3a`](https://github.com/vllm-project/vllm/commit/568afb3a13806beb53bb2e6bd518269357b237c0)
  and its [MXFP4 packed-dimension logic](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py#L73-L76)
- [Safetensors format](https://github.com/huggingface/safetensors#format)

*This is an independent technical estimate, not an official FLOPs disclosure.*
