"""Closed-loop toy autoregressive video-diffusion world (hidden test copy).

The benchmark generates short "videos" chunk by chunk, the way autoregressive
video diffusion models such as Self Forcing do:

* a video is ``chunks`` chunks of ``FRAMES_PER_CHUNK`` frames; every frame is a
  ``GRID x GRID`` token grid, tokens are ordered so that each ``BLOCK_SIDE x
  BLOCK_SIDE`` spatial tile is one contiguous attention block;
* each chunk is produced by a 4-step denoising loop (noise levels ``SIGMAS``)
  whose denoiser is one multi-head attention layer over the KV cache of all
  previously generated (clean) chunks plus the current noisy chunk;
* the finished chunk is appended to the KV cache, so attention errors in an
  early chunk are inherited by every later chunk (closed loop).

The scene is seen by a moving camera that pans, turns and sometimes cuts back
to a place it filmed long ago. Query/key features combine fine and coarse
world-position codes, a temporal code, a first-chunk "sink" code, a noise-level
code and the token content, each with per-instance random parameters and a
random rotation per head. Heads therefore show the patterns sparse attention
for AR video has to handle: local/diagonal, long-range retrieval of revisited
places, attention sinks and content matching.

Nothing here knows about any sparse-attention method. A method only chooses
block masks; the evaluator computes every attention output, every rollout and
every metric itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

GRID = 16
BLOCK_SIDE = 4
TOKENS_PER_FRAME = GRID * GRID
BLOCK_TOKENS = BLOCK_SIDE * BLOCK_SIDE
BLOCKS_PER_FRAME = TOKENS_PER_FRAME // BLOCK_TOKENS
FRAMES_PER_CHUNK = 3
HEADS = 4
HEAD_DIM = 64
CONTENT_DIM = 16
FINE_FREQS = 16
COARSE_FREQS = 4
TIME_OMEGAS = (math.pi / 16.0, math.pi / 40.0)
SIGMAS = (1.0, 0.75, 0.5, 0.25)
PROBE_COST = 0.5
RECENT_WINDOW_FRAMES = 6
ROLES = ("local", "global", "sink", "content")
DTYPE = torch.float64

DEFAULT_PRESET: dict = {
    "speed": (0.4, 1.0),
    "turn_std": 0.35,
    "revisits_per_frame": 0.06,
    "revisit_jitter": 0.75,
    "sharpness": 1.0,
}


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def local_offsets() -> np.ndarray:
    """Token (y, x) offsets from the frame centre, in block-major token order."""

    nb = GRID // BLOCK_SIDE
    offs = np.zeros((TOKENS_PER_FRAME, 2))
    idx = 0
    for b in range(BLOCKS_PER_FRAME):
        bi, bj = divmod(b, nb)
        for t in range(BLOCK_TOKENS):
            ii, jj = divmod(t, BLOCK_SIDE)
            offs[idx] = (
                bi * BLOCK_SIDE + ii - (GRID - 1) / 2.0,
                bj * BLOCK_SIDE + jj - (GRID - 1) / 2.0,
            )
            idx += 1
    return offs


def camera_path(rng: np.random.Generator, frames: int, preset: dict) -> np.ndarray:
    """Pan/turn camera with forced cut-backs to places filmed long ago."""

    window = RECENT_WINDOW_FRAMES
    earliest = window + FRAMES_PER_CHUNK + 1
    candidates = list(range(earliest, frames))
    revisit_frames: set[int] = set()
    if candidates:
        n_rev = max(1, int(rng.poisson(preset["revisits_per_frame"] * len(candidates))))
        n_rev = min(n_rev, len(candidates))
        revisit_frames = set(int(t) for t in rng.choice(candidates, size=n_rev, replace=False))

    path = np.zeros((frames, 2))
    pos = np.zeros(2)
    heading = _u(rng, 0.0, 2.0 * math.pi)
    speed = _u(rng, *preset["speed"])
    for t in range(1, frames):
        if t in revisit_frames:
            best: tuple[float, np.ndarray] | None = None
            recent = path[max(0, t - window):t]
            for _ in range(64):
                src = int(rng.integers(0, t - window - 1))
                cand = path[src] + rng.normal(0.0, preset["revisit_jitter"], 2)
                dist = float(np.abs(recent - cand).max(axis=1).min())
                if best is None or dist > best[0]:
                    best = (dist, cand)
                if dist >= GRID:
                    break
            assert best is not None
            pos = best[1].copy()
            heading = _u(rng, 0.0, 2.0 * math.pi)
        else:
            heading += float(rng.normal(0.0, preset["turn_std"]))
            pos = pos + speed * np.array([math.sin(heading), math.cos(heading)])
        path[t] = pos
    return path


def _random_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))[None, :]


def _head_params(rng: np.random.Generator, role: str, sharp: float) -> dict:
    if role == "local":
        p = dict(beta_f=_u(rng, 7, 10), ell_f=_u(rng, 0.8, 1.2), beta_c=_u(rng, 1, 2),
                 ell_c=_u(rng, 6, 10), time=(_u(rng, 6, 10), _u(rng, 8, 14)), sink=0.0,
                 gamma=_u(rng, 0, 1), mix=0.35)
    elif role == "global":
        p = dict(beta_f=_u(rng, 7, 10), ell_f=_u(rng, 1.0, 1.5), beta_c=_u(rng, 2, 4),
                 ell_c=_u(rng, 6, 10), time=(_u(rng, 0, 1), _u(rng, 0.5, 2)), sink=0.0,
                 gamma=_u(rng, 0, 1), mix=0.30)
    elif role == "sink":
        p = dict(beta_f=_u(rng, 1, 3), ell_f=_u(rng, 2, 4), beta_c=_u(rng, 1, 3),
                 ell_c=_u(rng, 6, 10), time=(_u(rng, 0, 1), _u(rng, 0, 1)), sink=_u(rng, 3, 5),
                 gamma=0.0, mix=0.10)
    elif role == "content":
        p = dict(beta_f=_u(rng, 1, 3), ell_f=_u(rng, 1.5, 3), beta_c=_u(rng, 0, 1),
                 ell_c=_u(rng, 6, 10), time=(_u(rng, 0, 1), _u(rng, 1, 4)), sink=0.0,
                 gamma=_u(rng, 5, 8), mix=0.25)
    else:  # pragma: no cover
        raise ValueError(role)
    p["beta_f"] *= sharp
    p["beta_c"] *= sharp
    p["noise_pen"] = _u(rng, 3, 5)
    p["mix"] *= _u(rng, 0.8, 1.2)
    p["role"] = role
    return p


@dataclass
class World:
    """One random scene, camera path, attention layer and noise sequence."""

    seed: int
    chunks: int
    preset: dict = field(default_factory=lambda: dict(DEFAULT_PRESET))

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.frames = self.chunks * FRAMES_PER_CHUNK
        self.path = camera_path(rng, self.frames, self.preset)
        offs = local_offsets()
        self.wpos = self.path[:, None, :] + offs[None, :, :]  # [F, T, 2]

        roles = list(ROLES)
        rng.shuffle(roles)
        sharp = float(self.preset.get("sharpness", 1.0))
        self.heads = [_head_params(rng, role, sharp) for role in roles]
        mix = np.array([h["mix"] for h in self.heads])
        self.mix = torch.tensor(mix / mix.sum(), dtype=DTYPE)
        self.kappa = _u(rng, 0.8, 1.2)

        pos_feats = []
        for h in self.heads:
            om_f = rng.normal(size=(FINE_FREQS, 2)) / h["ell_f"]
            om_c = rng.normal(size=(COARSE_FREQS, 2)) / h["ell_c"]
            ang_f = self.wpos @ om_f.T  # [F, T, P]
            ang_c = self.wpos @ om_c.T
            feat = np.concatenate(
                [
                    math.sqrt(h["beta_f"] / FINE_FREQS) * np.cos(ang_f),
                    math.sqrt(h["beta_f"] / FINE_FREQS) * np.sin(ang_f),
                    math.sqrt(h["beta_c"] / COARSE_FREQS) * np.cos(ang_c),
                    math.sqrt(h["beta_c"] / COARSE_FREQS) * np.sin(ang_c),
                ],
                axis=-1,
            )
            pos_feats.append(feat)
        self.pos_feat = torch.tensor(np.stack(pos_feats, axis=1), dtype=DTYPE)  # [F, H, T, Ppos]
        self.time_amp = torch.tensor(
            [[math.sqrt(a) for a in h["time"]] for h in self.heads], dtype=DTYPE
        )
        self.sink = torch.tensor([math.sqrt(h["sink"]) for h in self.heads], dtype=DTYPE)
        self.noise_pen = torch.tensor([math.sqrt(h["noise_pen"]) for h in self.heads], dtype=DTYPE)
        self.gamma = torch.tensor(
            [math.sqrt(h["gamma"] / CONTENT_DIM) for h in self.heads], dtype=DTYPE
        )
        self.rot = torch.tensor(
            np.stack([_random_rotation(rng, HEAD_DIM) for _ in self.heads]), dtype=DTYPE
        )
        n_q = FRAMES_PER_CHUNK * TOKENS_PER_FRAME
        self.eps = torch.tensor(
            rng.normal(size=(self.chunks, len(SIGMAS), n_q, CONTENT_DIM)), dtype=DTYPE
        )

    # ------------------------------------------------------------------ features
    def features(self, frame_ids: list[int], content: torch.Tensor, sigma: float, query: bool) -> torch.Tensor:
        """Per-head query or key features [H, N, HEAD_DIM] of the given frames."""

        nf = len(frame_ids)
        n = nf * TOKENS_PER_FRAME
        pos = self.pos_feat[frame_ids]  # [nf, H, T, P]
        pos = pos.permute(1, 0, 2, 3).reshape(HEADS, n, -1)
        t = torch.tensor(frame_ids, dtype=DTYPE).repeat_interleave(TOKENS_PER_FRAME)
        time = []
        for k, om in enumerate(TIME_OMEGAS):
            amp = self.time_amp[:, k][:, None]
            time.append(amp * torch.cos(om * t)[None, :])
            time.append(amp * torch.sin(om * t)[None, :])
        time_feat = torch.stack(time, dim=-1)  # [H, N, 4]
        ones = torch.ones(HEADS, n, 1, dtype=DTYPE)
        if query:
            sink = self.sink[:, None, None] * ones
            noise = -self.noise_pen[:, None, None] * ones
        else:
            first = (t < FRAMES_PER_CHUNK).to(DTYPE)[None, :, None]
            sink = self.sink[:, None, None] * first
            noise = self.noise_pen[:, None, None] * float(sigma) * ones
        cont = self.gamma[:, None, None] * content[None, :, :]
        raw = torch.cat([pos, time_feat, sink, noise, cont], dim=-1)
        pad = HEAD_DIM - raw.shape[-1]
        if pad > 0:
            raw = torch.cat([raw, torch.zeros(HEADS, n, pad, dtype=DTYPE)], dim=-1)
        rotated = torch.einsum("hnd,hed->hne", raw, self.rot)
        return rotated * HEAD_DIM ** 0.25

    # --------------------------------------------------------------- bookkeeping
    def kv_blocks(self, chunk: int) -> int:
        return (chunk + 1) * FRAMES_PER_CHUNK * BLOCKS_PER_FRAME

    def revisit_tokens(self, chunk: int) -> np.ndarray:
        """Tokens of ``chunk`` showing a place filmed before but not in the recent window."""

        t0 = chunk * FRAMES_PER_CHUNK
        w = self.wpos[t0:t0 + FRAMES_PER_CHUNK].reshape(-1, 2)
        if t0 == 0:
            return np.zeros(w.shape[0], dtype=bool)
        half = (GRID - 1) / 2.0 + 0.5
        dist = np.abs(w[:, None, :] - self.path[None, :t0, :]).max(axis=-1)  # [N, t0]
        in_view = dist <= half
        cut = max(0, t0 - RECENT_WINDOW_FRAMES)
        old = in_view[:, :cut].any(axis=1)
        recent = in_view[:, cut:].any(axis=1)
        return old & ~recent


def budget_tiles(chunks: int, target_sparsity: float) -> tuple[float, float]:
    """Tile budget of one rollout and its dense tile count.

    The first chunk may be dense; every later chunk gets ``1 - target_sparsity``
    of its dense tiles. Methods may redistribute the total however they like.
    """

    nqb = FRAMES_PER_CHUNK * BLOCKS_PER_FRAME
    per_step = HEADS * len(SIGMAS) * nqb
    kvb = [(i + 1) * FRAMES_PER_CHUNK * BLOCKS_PER_FRAME for i in range(chunks)]
    dense = per_step * float(sum(kvb))
    budget = per_step * (kvb[0] + (1.0 - target_sparsity) * float(sum(kvb[1:])))
    return budget, dense


class BudgetExceeded(RuntimeError):
    pass


class Call:
    """One attention call handed to a mask selector."""

    def __init__(self, chunk: int, step: int, sigma: float, q: torch.Tensor, k: torch.Tensor,
                 logits: torch.Tensor, rollout: "RolloutState") -> None:
        self.chunk = chunk
        self.step = step
        self.sigma = sigma
        self.frames_cached = (chunk + 1) * FRAMES_PER_CHUNK
        self.current_frame_start = chunk * FRAMES_PER_CHUNK
        self.nqb = q.shape[1] // BLOCK_TOKENS
        self.nkb = k.shape[1] // BLOCK_TOKENS
        self.q_pooled = q.reshape(HEADS, self.nqb, BLOCK_TOKENS, HEAD_DIM).mean(2)
        self.k_pooled = k.reshape(HEADS, self.nkb, BLOCK_TOKENS, HEAD_DIM).mean(2)
        self._logits = logits
        self._rollout = rollout

    def tile_lse(self) -> torch.Tensor:
        """Exact log-sum-exp of every tile's logits [H, nqb, nkb] (evaluator side only)."""

        view = self._logits.reshape(HEADS, self.nqb, BLOCK_TOKENS, self.nkb, BLOCK_TOKENS)
        return torch.logsumexp(view, dim=(2, 4))

    def probe(self, tiles: np.ndarray) -> np.ndarray:
        """Exact tile log-sum-exp for the requested tiles; costs PROBE_COST per tile."""

        tiles = np.asarray(tiles, dtype=bool)
        self._rollout.charge(PROBE_COST * float(tiles.sum()))
        lse = self.tile_lse().numpy().astype(np.float32)
        out = np.full(lse.shape, np.nan, dtype=np.float32)
        out[tiles] = lse[tiles]
        return out


class RolloutState:
    def __init__(self, budget: float | None) -> None:
        self.budget = budget
        self.cost = 0.0

    def charge(self, amount: float) -> None:
        self.cost += amount
        if self.budget is not None and self.cost > self.budget + 1e-6:
            raise BudgetExceeded(f"tile cost {self.cost:.1f} exceeds budget {self.budget:.1f}")


def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-12)


def rollout(world: World, select, budget: float | None = None) -> dict:
    """Generate the whole video with masks from ``select(call) -> bool [H, nqb, nkb]``.

    ``select=None`` gives the dense reference rollout.
    """

    state = RolloutState(budget)
    f = FRAMES_PER_CHUNK
    past_k: torch.Tensor | None = None
    past_v: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []
    attn_cos: list[float] = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for i in range(world.chunks):
        frames = list(range(i * f, (i + 1) * f))
        x = world.eps[i, 0].clone()
        x0 = x
        for j, sigma in enumerate(SIGMAS):
            q = world.features(frames, x, sigma, query=True)
            kc = world.features(frames, x, sigma, query=False)
            k = kc if past_k is None else torch.cat([past_k, kc], dim=1)
            v = x if past_v is None else torch.cat([past_v, x], dim=0)
            logits = torch.einsum("hqd,hkd->hqk", q, k) * scale
            m = logits.amax(dim=-1, keepdim=True)
            e = torch.exp(logits - m)
            dense = (e @ v) / e.sum(dim=-1, keepdim=True)  # [H, nq, C]
            if select is None:
                att = dense
            else:
                call = Call(i, j, sigma, q, k, logits, state)
                mask = np.asarray(select(call), dtype=bool)
                state.charge(float(mask.sum()))
                mt = torch.from_numpy(mask).to(DTYPE)
                nqb, nkb = call.nqb, call.nkb
                es = (e.reshape(HEADS, nqb, BLOCK_TOKENS, nkb, BLOCK_TOKENS)
                      * mt[:, :, None, :, None]).reshape(e.shape)
                z = es.sum(dim=-1, keepdim=True)
                att = torch.where(z > 0, (es @ v) / z.clamp_min(1e-300), torch.zeros_like(dense))
            mixed_sparse = torch.einsum("h,hnc->nc", world.mix, att)
            mixed_dense = torch.einsum("h,hnc->nc", world.mix, dense)
            denom = mixed_sparse.norm() * mixed_dense.norm()
            attn_cos.append(float((mixed_sparse * mixed_dense).sum() / denom) if denom > 0 else 0.0)
            skip = world.kappa * (1.0 - sigma)
            x0 = _rmsnorm(mixed_sparse + skip * x)
            if j + 1 < len(SIGMAS):
                nxt = SIGMAS[j + 1]
                x = (1.0 - nxt) * x0 + nxt * world.eps[i, j + 1]
        outputs.append(x0)
        kclean = world.features(frames, x0, 0.0, query=False)
        past_k = kclean if past_k is None else torch.cat([past_k, kclean], dim=1)
        past_v = x0 if past_v is None else torch.cat([past_v, x0], dim=0)
    return {"outputs": outputs, "attn_cos": attn_cos, "cost": state.cost}


def rollout_metrics(world: World, run: dict, dense_run: dict, dense_tiles: float) -> dict:
    """Per-rollout metrics against the dense rollout of the same world."""

    errs = []
    num = 0.0
    den = 0.0
    for i, (got, ref) in enumerate(zip(run["outputs"], dense_run["outputs"])):
        diff = got - ref
        errs.append(float(diff.norm() / ref.norm()))
        rv = torch.from_numpy(world.revisit_tokens(i))
        if rv.any():
            num += float((diff[rv] ** 2).sum())
            den += float((ref[rv] ** 2).sum())
    return {
        "mean_generation_rel_error": float(np.mean(errs)),
        "final_chunk_rel_error": errs[-1],
        "revisit_rel_error": math.sqrt(num / den) if den > 0 else float("nan"),
        "mean_attention_cosine": float(np.mean(run["attn_cos"])),
        "tile_cost_fraction": run["cost"] / dense_tiles,
        "chunk_errors": errs,
    }
