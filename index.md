# Blogs

Notes on LLM inference, training, and systems performance — written while
learning, corrections welcome.

## Posts

| Date | Post | Topics |
|---|---|---|
| 2026-09 | [Fast Model Switching with a Shared Host-Memory Weight Cache](model_switch.md) | Model serving, shared memory, CUDA, weight loading |
| 2026-09 | [WIP: How an Early NCCL Receive Turned CUDA Lazy Loading Into Pipeline Bubbles](wip-how-an-early-nccl-receive-created-pipeline-bubbles.md) | CUDA lazy loading, NCCL, pipeline parallelism, Nsight Systems |
| 2026-09 | [How to Count Inference FLOPs from a Quantized Checkpoint](counting-inference-flops-from-a-quantized-checkpoint.md) *(WIP)* | FLOPs accounting, MFU, MXFP4 quantization, sparse MoE, safetensors |

## About the math notation

Posts use `$$ ... $$` LaTeX blocks. These render natively when viewing the
markdown on github.com. If this repo is later served through GitHub Pages,
add a KaTeX auto-render include to the page layout.
