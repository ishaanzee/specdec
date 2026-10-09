"""Rollback-capable gated-delta-net (linear attention) layers.

Why this exists
---------------
Qwen3.8 (mlx-lm model type ``qwen3_5``) is a hybrid: 48 of its 64 layers are
gated delta nets (GDN, linear attention), the other 16 are softmax attention
with a normal KV cache. A GDN layer carries two pieces of state from token to
token, both stored in an ``ArraysCache(size=2)`` by mlx-lm:

* ``cache[0]``: the causal-conv window, the last ``kernel_size - 1 = 3`` rows
  of the ``in_proj_qkv`` output, shape ``[B, 3, conv_dim]`` (bf16).
* ``cache[1]``: the recurrent state ``S``, shape ``[B, Hv, Dv, Dk]`` =
  ``[1, 48, 128, 128]`` float32 = 3.1 MB per layer, 151 MB for all 48 layers.

When the target verifies ``T = 1 + k`` tokens in one forward pass, the KV
cache of an attention layer can be trimmed afterwards, but the recurrent state
``S`` has absorbed every one of the ``T`` tokens and there is no way to
subtract the rejected ones out of it. To roll back to "after ``n_keep``
tokens" we need ``S_{n_keep}``, which the stock layer never materialises.

This module replaces each GDN layer's forward (``RollbackGDN``) so that, when
it is given a ``LinearCache``, it records what is needed to recover any
intermediate state. ``LinearCache.commit(n_keep)`` then rewinds the layer.
Three strategies are implemented (``LinearCache.mode``):

``"replay"`` (default)
    Record the per-token recurrence inputs ``q, k, v, g, beta`` (a few hundred
    KB per layer) and the pre-verify state ``S_0``. On partial acceptance,
    rerun only the recurrence kernel for the ``n_keep`` accepted tokens
    starting from ``S_0``. No projections, no MLPs, no attention: one read and
    one write of ``S`` per layer (~2 x 151 MB). It uses the very same Metal
    kernel as the verify pass, whose per-token arithmetic does not depend on
    ``T``, so the result is bitwise identical to the state the verify pass
    held in registers after token ``n_keep``.

``"capture"``
    A variant of mlx-lm's packed GDN kernel (``_capture_kernel``) that writes
    the state after *every* token to an extra output ``[B, T, Hv, Dv, Dk]``;
    commit just indexes it. No recompute at all, but the verify pass now
    writes ``T x 151 MB``, which on a ~150 GB/s machine costs about 1 ms per
    verified token whether or not anything is rejected.

``"reforward"``
    The naive baseline: snapshot ``(conv, S)`` before verify; on partial
    acceptance restore it, trim the KV caches to the pre-verify length and run
    the whole model again over the accepted tokens. Implemented in
    ``SpecModel.commit`` because it needs the full model.

The conv window is cheap in all modes: the layer keeps the concatenated
``[conv_state, qkv]`` rows of the verify pass, and the window after
``n_keep`` tokens is simply rows ``n_keep : n_keep + 3`` of it.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk

ROLLBACK_MODES = ("replay", "capture", "reforward")


def _make_capture_kernel():
    """mlx-lm's packed scalar-gate Dk=128 GDN kernel, plus per-token states.

    The body is copied from ``mlx_lm.models.gated_delta._make_gated_delta_packed_kernel``
    with one change: instead of writing the final state once, it writes the
    state after every token to ``states[b, t]``. The arithmetic per token is
    unchanged, so ``states[:, t]`` is bitwise what the stock kernel would
    return as its final state for the first ``t + 1`` tokens, and
    ``states[:, T - 1]`` is the usual final state.
    """
    source = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        // states: [B, T, Hv, Dv, Dk]; advance by one token = Hv * Dv * Dk
        auto o_states = states + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk
                        + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          for (int i = 0; i < values_per_lane; ++i) {
            o_states[i] = static_cast<StT>(state[i]);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
          o_states += Hv * Dv * Dk;
        }
    """
    return mx.fast.metal_kernel(
        name="specdec_gated_delta_capture",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


_capture_kernel = _make_capture_kernel() if mx.metal.is_available() else None


def gated_delta_capture(q, k, v, g, beta, state):
    """Run the GDN recurrence and return ``(y, states)``, states per token.

    Shapes: q, k ``[B, T, Hk, Dk]``; v ``[B, T, Hv, Dv]``; g, beta
    ``[B, T, Hv]`` (scalar gates); state ``[B, Hv, Dv, Dk]`` float32.
    Returns y ``[B, T, Hv, Dv]`` and states ``[B, T, Hv, Dv, Dk]``.
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if not (g.ndim == 3 and Dk == 128 and Dv % 8 == 0
            and g.dtype == mx.float32 and state.dtype == mx.float32):
        raise ValueError("capture kernel only supports the packed-kernel shapes")
    return _capture_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk),
                  ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv // 8, B * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[q.dtype, state.dtype],
    )


class LinearCache(ArraysCache):
    """``ArraysCache(size=2)`` for one GDN layer that can be rewound.

    ``self[0]`` / ``self[1]`` keep mlx-lm's meaning (conv window, recurrent
    state). After a forward pass through ``RollbackGDN`` the cache also holds
    ``self.pending``, the information needed to rewind into the middle of that
    pass. ``commit(n_keep)`` keeps the first ``n_keep`` tokens of the last
    pass and forgets the rest; it must be called after every pass (also when
    everything is kept, which just drops ``pending``).
    """

    def __init__(self, mode: str = "replay"):
        super().__init__(size=2)
        if mode not in ROLLBACK_MODES:
            raise ValueError(f"unknown rollback mode {mode!r}")
        self.mode = mode
        self.pending: Optional[dict] = None

    def record(self, conv_input, T, state0, final_state, **recurrence):
        """Store the pass's rollback info and advance to the end of the pass."""
        self.pending = dict(conv_input=conv_input, state0=state0, T=T, **recurrence)
        n_conv = conv_input.shape[1] - T
        self[0] = mx.contiguous(conv_input[:, -n_conv:, :])
        self[1] = final_state

    def commit(self, n_keep: int):
        p, self.pending = self.pending, None
        if p is None or n_keep == p["T"]:
            return
        n_conv = p["conv_input"].shape[1] - p["T"]
        # Rows [n_keep, n_keep + n_conv) are the last n_conv inputs once the
        # first n_keep tokens have been consumed (row 0..n_conv-1 = old window).
        self[0] = mx.contiguous(p["conv_input"][:, n_keep : n_keep + n_conv, :])
        if n_keep == 0:
            self[1] = p["state0"]
        elif self.mode == "capture":
            self[1] = p["states"][:, n_keep - 1]
        elif self.mode == "replay":
            r = {name: p[name][:, :n_keep] for name in ("q", "k", "v", "g", "beta")}
            _, self[1] = gated_delta_kernel(**r, state=p["state0"])
        else:  # reforward: restore the pre-pass state; SpecModel reruns the tokens
            self[0] = mx.contiguous(p["conv_input"][:, :n_conv, :])
            self[1] = p["state0"]


class RollbackGDN(nn.Module):
    """Wraps an mlx-lm ``qwen3_5.GatedDeltaNet``.

    With a plain ``ArraysCache`` it simply calls the wrapped layer, so the
    model behaves exactly as stock mlx-lm. With a ``LinearCache`` it runs the
    same computation (same ops, same kernel), batch size 1 and no padding
    mask, and records rollback information in the cache.
    """

    def __init__(self, gdn: nn.Module):
        super().__init__()
        self.gdn = gdn

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        if not isinstance(cache, LinearCache):
            return self.gdn(x, mask, cache)
        assert mask is None and x.shape[0] == 1, "spec path is batch 1, unpadded"
        L = self.gdn
        B, S, _ = x.shape

        qkv = L.in_proj_qkv(x)
        z = L.in_proj_z(x).reshape(B, S, L.num_v_heads, L.head_v_dim)
        b = L.in_proj_b(x)
        a = L.in_proj_a(x)

        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros((B, L.conv_kernel_size - 1, L.conv_dim), x.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = nn.silu(L.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [L.key_dim, 2 * L.key_dim], -1),
                [L.num_k_heads, L.num_k_heads, L.num_v_heads],
                [L.head_k_dim, L.head_k_dim, L.head_v_dim],
            )
        ]
        q, k = normalize_qk(q, k, inv_scale=L.head_k_dim**-0.5, eps=1e-6)
        # Same as mlx-lm's gated_delta_update, unrolled so we can keep g / beta.
        beta = mx.sigmoid(b)
        g = compute_g(L.A_log, a, L.dt_bias)
        state0 = cache[1]
        if state0 is None:
            state0 = mx.zeros((B, L.num_v_heads, L.head_v_dim, L.head_k_dim), mx.float32)

        if cache.mode == "capture":
            out, states = gated_delta_capture(q, k, v, g, beta, state0)
            cache.record(conv_input, S, state0, states[:, -1], states=states)
        else:
            out, state = gated_delta_kernel(q, k, v, g, beta, state0)
            cache.record(conv_input, S, state0, state, q=q, k=k, v=v, g=g, beta=beta)

        out = L.norm(out, z)
        return L.out_proj(out.reshape(B, S, -1))
