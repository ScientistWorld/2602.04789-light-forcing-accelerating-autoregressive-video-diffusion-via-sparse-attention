"""Helpers shared by the red-team methods.

Everything here uses only what a future agent receives: the input generator in
data/, the reference method in method/ (improve mode) and baseline/. Nothing in
this file imports eval/.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path("/home/user")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from data.attention_inputs import make_structured_qkv  # noqa: E402

BLOCK = 64
OUTPUTS = ROOT / "outputs"
EXPERIMENTS: dict[str, dict] = {
    "scaled_attention_core": {
        "suffix": "",
        "chunks": 7,
        "frames_per_chunk": 3,
        "tokens_per_frame": 256,
        "dim": 64,
        "seed": 260204789,
    },
    "scaled_attention_long_context": {
        "suffix": "_long_context",
        "chunks": 9,
        "frames_per_chunk": 3,
        "tokens_per_frame": 256,
        "dim": 64,
        "seed": 260204790,
    },
}


def inputs(spec: dict, seed: int | None = None):
    return make_structured_qkv(
        chunks=spec["chunks"],
        frames_per_chunk=spec["frames_per_chunk"],
        tokens_per_frame=spec["tokens_per_frame"],
        dim=spec["dim"],
        seed=spec["seed"] if seed is None else seed,
    )


def history(k_chunks, v_chunks, i: int):
    return torch.cat(k_chunks[: i + 1], dim=0), torch.cat(v_chunks[: i + 1], dim=0)


def dense_outputs(q_chunks, k_chunks, v_chunks):
    """Full causal chunk attention: the quantity sparse methods approximate."""

    outs = []
    past_k, past_v = [], []
    for q, k, v in zip(q_chunks, k_chunks, v_chunks):
        past_k.append(k)
        past_v.append(v)
        keys = torch.cat(past_k, dim=0)
        values = torch.cat(past_v, dim=0)
        scores = q @ keys.T / math.sqrt(q.shape[-1])
        outs.append(torch.softmax(scores, dim=-1) @ values)
    return outs


def block_mean(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    n = math.ceil(x.shape[0] / block)
    pad = torch.zeros(n * block, x.shape[1], dtype=x.dtype)
    pad[: x.shape[0]] = x
    return pad.view(n, block, x.shape[1]).mean(dim=1)


def block_attention(q, k, v, mask, block: int = BLOCK):
    """Honest blockwise sparse attention: only tiles with mask=True contribute."""

    token_mask = mask.bool().repeat_interleave(block, 0).repeat_interleave(block, 1)
    token_mask = token_mask[: q.shape[0], : k.shape[0]]
    scores = (q @ k.T / math.sqrt(q.shape[-1])).masked_fill(~token_mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def artifact_path(name: str, experiment: str) -> Path:
    return OUTPUTS / f"{name}{EXPERIMENTS[experiment]['suffix']}.pt"


def save_artifact(name, experiment, outputs, masks=None, stats=None, keep_stats=False):
    art = {
        "outputs": [o.detach().cpu().float() for o in outputs],
        "stats": dict(stats or {}),
    }
    if masks is not None:
        art["masks"] = [m.detach().cpu() for m in masks]
        if not keep_stats:
            art["stats"]["active_blocks"] = float(sum(float(m.float().sum()) for m in masks))
            art["stats"]["total_blocks"] = float(sum(m.numel() for m in masks))
    path = artifact_path(name, experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    print(f"wrote {path}")
    return path


def cag_sparsities(spec: dict, s_target: float = 0.90, s_base: float = 0.98):
    from method.light_forcing_attention import allocate_chunk_sparsities

    return allocate_chunk_sparsities(
        chunks=spec["chunks"],
        frames_per_chunk=spec["frames_per_chunk"],
        tokens_per_frame=spec["tokens_per_frame"],
        block_size=BLOCK,
        s_target=s_target,
        s_base=s_base,
    )


def run_reference(spec: dict, seed: int | None = None, **overrides):
    """The gym's reference method (method/light_forcing_attention.py)."""

    from method.light_forcing_attention import LightForcingConfig, run_light_forcing

    kwargs = dict(
        chunks=spec["chunks"],
        frames_per_chunk=spec["frames_per_chunk"],
        tokens_per_frame=spec["tokens_per_frame"],
        dim=spec["dim"],
        seed=spec["seed"] if seed is None else seed,
    )
    kwargs.update(overrides)
    return run_light_forcing(LightForcingConfig(**kwargs))


# --------------------------------------------------------------------------
# Block scorers for the selection-variant family. Each returns a
# (query_blocks, kv_blocks) tensor; higher means "keep this tile".
# --------------------------------------------------------------------------


def score_mean_pool(q, k, qb, kb, gen=None):
    """Paper's block logits: pooled query block vs pooled key block."""

    return qb @ kb.T


def score_lse(q, k, qb, kb, gen=None):
    """Pooled query block vs every key token, log-sum-exp within each key block."""

    nqb, nkb = qb.shape[0], kb.shape[0]
    s = (qb @ k.T) / math.sqrt(q.shape[-1])
    return s.view(nqb, nkb, BLOCK).logsumexp(dim=-1)


def score_sampled_rows(q, k, qb, kb, gen=None, stride: int = 8):
    """Exact softmax mass of every `stride`-th query row, summed per tile."""

    nqb, nkb = qb.shape[0], kb.shape[0]
    rows = q.view(nqb, BLOCK, q.shape[-1])[:, ::stride, :]
    p = torch.softmax(rows @ k.T / math.sqrt(q.shape[-1]), dim=-1)
    return p.view(nqb, rows.shape[1], nkb, BLOCK).sum(dim=(1, 3))


def score_oracle(q, k, qb, kb, gen=None):
    """Brute force: exact dense softmax mass per tile (needs full QK^T)."""

    nqb, nkb = qb.shape[0], kb.shape[0]
    p = torch.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1)
    return p.view(nqb, BLOCK, nkb, BLOCK).sum(dim=(1, 3))


def score_random(q, k, qb, kb, gen=None):
    """Key idea removed: tiles chosen uniformly at random."""

    return torch.rand(qb.shape[0], kb.shape[0], generator=gen)


def score_recency(q, k, qb, kb, gen=None):
    """Key idea removed: most recent tiles only (local window, no retrieval)."""

    return torch.arange(kb.shape[0], dtype=torch.float32).expand(qb.shape[0], kb.shape[0]).clone()


def selection_attention(q_chunks, k_chunks, v_chunks, spec, sparsities, scorer,
                        topk_frames: int = 6, use_frames: bool = True, seed: int = 0):
    """Dense first chunk, then per-query-block top-N tiles.

    N per chunk follows `sparsities` (same rule as the reference). With
    use_frames=True the candidate tiles are restricted to the reference's
    frame retrieval (top-k past frames by pooled logits + current frames).
    """

    gen = torch.Generator().manual_seed(seed)
    fpc = spec["frames_per_chunk"]
    tpf = spec["tokens_per_frame"]
    bpf = math.ceil(tpf / BLOCK)
    outs, masks = [], []
    for i, q in enumerate(q_chunks):
        k, v = history(k_chunks, v_chunks, i)
        nqb = math.ceil(q.shape[0] / BLOCK)
        nkb = math.ceil(k.shape[0] / BLOCK)
        if i == 0:
            mask = torch.ones(nqb, nkb, dtype=torch.bool)
        else:
            sel = max(1, int(round((1.0 - sparsities[i]) * nkb)))
            qb = block_mean(q)
            kb = block_mean(k)
            cur = i * fpc
            past_frame_keys = kb[: cur * bpf].view(cur, bpf, -1).mean(dim=1)
            scores = scorer(q, k, qb, kb, gen)
            mask = torch.zeros(nqb, nkb, dtype=torch.bool)
            for r in range(nqb):
                allowed = torch.zeros(nkb, dtype=torch.bool)
                if use_frames:
                    top = min(topk_frames, cur)
                    frames = torch.topk(qb[r] @ past_frame_keys.T, k=top).indices.tolist() if top else []
                    frames += list(range(cur, cur + fpc))
                    for f in frames:
                        allowed[f * bpf:(f + 1) * bpf] = True
                else:
                    allowed[:] = True
                s = scores[r].masked_fill(~allowed, float("-inf"))
                n = min(sel, int(allowed.sum()))
                mask[r, torch.topk(s, k=n).indices] = True
        outs.append(block_attention(q, k, v, mask))
        masks.append(mask)
    return outs, masks


SCORERS = {
    "mean": score_mean_pool,
    "lse": score_lse,
    "sampled": score_sampled_rows,
    "oracle": score_oracle,
    "random": score_random,
    "recency": score_recency,
}
