# Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening).

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2602.04789"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2602.04789-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2602.04789](https://arxiv.org/abs/2602.04789) |
|:---|:---|
| **Domain** | Video generation / efficient attention |
| **TL;DR** | The paper makes sparse attention practical for autoregressive video diffusion by allocating sparsity across chunks and selecting historical context hierarchically. |

## Summary

Autoregressive video diffusion can stream long videos, but attention over the growing history becomes the main runtime bottleneck. Light Forcing addresses this with chunk-aware sparsity allocation and coarse-to-fine sparse context retrieval. The paper reports VBench quality comparable to or better than dense attention with lower latency.

## Problem

How can autoregressive video generators reduce the quadratic cost of growing historical attention without losing the visual consistency and motion quality that come from using long context?

## The paper's method

Light Forcing is a sparse attention method tailored to autoregressive video diffusion. Its key idea is to assign different sparsity budgets to chunks and to retrieve historical context in a two-stage frame/block hierarchy.

## Key results of the paper

- Self Forcing 1.3B: VBench total score 84.5 with 1.30x end-to-end speedup.
- LongLive 1.3B: VBench total score 83.9 with 1.19x end-to-end speedup.
- Efficient deployment with FP8 and LightVAE reports 19.7 FPS on RTX 5090.

## Run it

```bash
pip install sciworld
sw research 2602.04789 --mode innovate --agent claude-code
sw evaluate 2602.04789
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

