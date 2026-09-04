"""Real-Time Chunking (RTC) inference for the JAX pi0 / pi0.5 flow-matching models.

RTC ("Real-Time Execution of Action Chunking Flow Policies", arXiv:2506.07339) turns
chunk-to-chunk handover into an inpainting problem: while denoising a new action chunk, a
soft-masked guidance term pulls the leading timesteps of the new chunk towards the
unexecuted tail of the previous chunk, so the robot never sees a discontinuity at the
switch point.

The implementation follows Physical Intelligence's own JAX reference,
https://github.com/Physical-Intelligence/real-time-chunking-kinetix
(`src/model.py`, `get_prefix_weights` and `Model.realtime_action`), adapted to openpi's
time convention.

Time conventions
----------------
PI's kinetix model runs flow time ``t: 0 -> 1`` (``t=0`` noise, ``t=1`` data) with
``dt = +1/num_steps`` and a velocity that points data-ward. openpi runs ``time: 1 -> 0``
(``time=1`` noise, ``time=0`` data) with ``dt = -1/num_steps`` and a velocity that points
noise-ward. The mapping is ``t = 1 - time`` and ``v_openpi = -v_kinetix``, which gives:

* clean-sample estimate: PI ``x_1 = x_t + (1 - t) v_t``  ->  openpi ``x_1 = x_t - time * v_t``
* guided velocity:       PI ``v_t + w * correction``     ->  openpi ``v_t - w * correction``

Both rewritings are exact; the two integrators take identical steps.
"""

import dataclasses
from typing import Literal, Protocol

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model

PrefixAttentionSchedule = Literal["zeros", "ones", "linear", "exp"]


def get_prefix_weights(
    start: int | jax.Array, end: int | jax.Array, total: int, schedule: PrefixAttentionSchedule
) -> jax.Array:
    """Soft prefix mask over the `total` timesteps of the chunk being generated.

    Verbatim port of PI's kinetix `get_prefix_weights`. With start=2, end=6, total=10:

        1  1  4/5 3/5 2/5 1/5 0  0  0  0
               ^              ^
             start           end

    `start` (inclusive) is where the chunk starts being allowed to change -- in practice the
    inference delay, i.e. the number of leading timesteps that are already committed to the
    robot and therefore must be reproduced exactly. `end` (exclusive) is where the chunk
    stops paying attention to the prefix at all, i.e. the execution horizon. If ``start ==
    0`` the whole chunk is free; if ``end == total`` the whole prefix is attended to.

    `end` takes precedence over `start`: if ``end < start`` then `start` is pushed down to
    `end`, so ``end == 0`` ignores the prefix entirely.

    `start` and `end` may be traced; only `total` and `schedule` must be concrete.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total, dtype=jnp.float32)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def guidance_weight(time: jax.Array | float, max_guidance_weight: float) -> jax.Array:
    """RTC guidance weight at openpi flow time `time` (1 = noise, 0 = data).

    PI/kinetix in its own time ``t = 1 - time``:
        inv_r2 = (t**2 + (1 - t)**2) / (1 - t)**2
        c      = nan_to_num((1 - t) / t, posinf=max_guidance_weight)
        w      = min(c * inv_r2, max_guidance_weight)
    """
    time = jnp.asarray(time, dtype=jnp.float32)
    t = 1.0 - time
    inv_r2 = (t**2 + (1.0 - t) ** 2) / ((1.0 - t) ** 2)
    c = jnp.nan_to_num((1.0 - t) / t, posinf=max_guidance_weight)
    return jnp.minimum(jnp.nan_to_num(c * inv_r2, posinf=max_guidance_weight), max_guidance_weight)


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Inference-time RTC knobs.

    For orientation: PI's kinetix eval defaults to schedule="exp", max_guidance_weight=5.0,
    num_flow_steps=5, inference_delay=0, execute_horizon=1; LeRobot's RTCConfig defaults to
    schedule=LINEAR, max_guidance_weight=10.0, execution_horizon=10.
    """

    # Under `jax.jit`, `execution_horizon`, `inference_delay` and `max_guidance_weight` may be
    # traced values and can change per call without triggering a recompile. The two that select
    # which computation runs -- `prefix_attention_schedule` and `use_vjp` -- must stay concrete.
    #
    # Soft-mask shape over the prefix.
    prefix_attention_schedule: PrefixAttentionSchedule = "exp"
    # Upper clamp on the guidance weight, which diverges as time -> 1 (pure noise).
    max_guidance_weight: float = 10.0
    # Timesteps of the chunk that overlap the previous chunk, i.e. `end` of the soft mask.
    execution_horizon: int | jax.Array = 25
    # Timesteps already committed to the robot while this chunk is being computed, i.e.
    # `start` of the soft mask; these are pinned hard to the previous chunk.
    inference_delay: int | jax.Array = 0
    # True  -> PI's pinv correction, a real vector-Jacobian product through the denoiser.
    # False -> first-order form in which d(x_1)/d(x_t) is taken to be the identity, so the
    #          correction collapses to the weighted error itself. This is what LeRobot's
    #          `RTCProcessor.denoise_step` actually computes; exposed here for comparison.
    use_vjp: bool = True


class _Denoiser(Protocol):
    def __call__(self, x_t: jax.Array, time: jax.Array) -> jax.Array: ...


def corrected_velocity(
    denoise: _Denoiser,
    x_t: jax.Array,
    time: jax.Array,
    prev_chunk: jax.Array,
    weights: jax.Array,
    *,
    max_guidance_weight: float,
    use_vjp: bool,
) -> jax.Array:
    """One RTC-guided velocity evaluation, in openpi's time convention.

    Args:
        denoise: maps (x_t, time) -> v_t, the unguided flow velocity.
        x_t: [b, ah, ad] current noisy chunk.
        time: scalar flow time in (0, 1].
        prev_chunk: [b, ah, ad] previous chunk's unexecuted tail, aligned to index 0 of the
            new chunk and zero-padded out to the full horizon.
        weights: [ah] soft prefix mask from `get_prefix_weights`.
        max_guidance_weight: clamp on the guidance weight.
        use_vjp: see `RTCConfig.use_vjp`.
    """

    def x1_fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        v = denoise(x, time)
        # Estimate of the clean sample implied by the velocity at (x, time).
        return x - time * v, v

    if use_vjp:
        x_1, vjp_fn, v_t = jax.vjp(x1_fn, x_t, has_aux=True)
        err = (prev_chunk - x_1) * weights[None, :, None]
        correction = vjp_fn(err.astype(x_1.dtype))[0]
    else:
        x_1, v_t = x1_fn(x_t)
        err = (prev_chunk - x_1) * weights[None, :, None]
        # d(x_1)/d(x_t) treated as the identity: the Jacobian of the denoiser is dropped.
        correction = err

    return v_t - guidance_weight(time, max_guidance_weight) * correction


def _make_denoiser(model, observation: _model.Observation):
    """Prefill the prefix KV cache and return a (x_t, time) -> v_t closure.

    Mirrors the body of `Pi0.sample_actions` exactly, so with no guidance the two agree
    bit-for-bit.
    """
    # Imported lazily so that `pi0.py` can delegate to this module without a cycle.
    from openpi.models.pi0 import make_attn_mask

    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask_2d = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask_2d, positions=positions)

    def denoise(x_t: jax.Array, time: jax.Array) -> jax.Array:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    return denoise


def sample_actions_rtc(
    model,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    prev_chunk_left_over: jax.Array | None = None,
    config: RTCConfig | None = None,
    num_steps: int = 10,
    noise: jax.Array | None = None,
    **overrides,
) -> _model.Actions:
    """Sample an action chunk with Real-Time Chunking guidance.

    Args:
        model: a `Pi0` instance (pi0 or pi0.5).
        rng: used only to draw `noise` when it is not supplied.
        observation: un-preprocessed observation, exactly as for `sample_actions`.
        prev_chunk_left_over: [ah_prev, ad] or [b, ah_prev, ad] -- the previous chunk's
            unexecuted tail in *normalized* action space, aligned so that index 0 is the
            timestep this new chunk's index 0 will occupy. Shorter than the horizon is fine
            (it is zero-padded, and the execution horizon is clamped to its length). `None`
            disables guidance, in which case this reduces exactly to `sample_actions`.
        config: RTC knobs; `overrides` (e.g. `inference_delay=5`) are applied on top.
        num_steps: flow-matching steps, as in `sample_actions`.
        noise: [b, ah, ad] initial noise; drawn from `rng` if omitted.

    Returns:
        [b, ah, ad] the denoised action chunk.
    """
    config = dataclasses.replace(config or RTCConfig(), **overrides)

    observation = _model.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    denoise = _make_denoiser(model, observation)
    # openpi convention: t=1 is noise, t=0 is the target distribution.
    dt = -1.0 / num_steps

    if prev_chunk_left_over is None:
        step_fn = lambda x_t, time: denoise(x_t, time)  # noqa: E731
    else:
        prev = jnp.asarray(prev_chunk_left_over, dtype=jnp.float32)
        if prev.ndim == 2:
            prev = prev[None]
        if prev.shape[0] == 1 and batch_size != 1:
            prev = jnp.broadcast_to(prev, (batch_size, *prev.shape[1:]))
        prev_len = prev.shape[1]
        # A previous tail shorter than the requested execution horizon has nothing left to
        # merge with beyond its own length, so clamp (matches LeRobot's behaviour). Done with
        # `jnp.minimum` rather than `min` so `execution_horizon` may be a traced value: in a
        # real RTC loop the measured delay and horizon change every control cycle, and forcing
        # them static would recompile the whole sampler on each new value.
        end = jnp.minimum(jnp.minimum(jnp.asarray(config.execution_horizon), prev_len), model.action_horizon)
        pad = [(0, 0), (0, max(model.action_horizon - prev_len, 0)), (0, max(model.action_dim - prev.shape[2], 0))]
        prev = jnp.pad(prev[:, : model.action_horizon, : model.action_dim], pad)
        weights = get_prefix_weights(
            config.inference_delay, end, model.action_horizon, config.prefix_attention_schedule
        )

        def step_fn(x_t, time):
            return corrected_velocity(
                denoise,
                x_t,
                time,
                prev,
                weights,
                max_guidance_weight=config.max_guidance_weight,
                use_vjp=config.use_vjp,
            )

    def step(carry):
        x_t, time = carry
        return x_t + dt * step_fn(x_t, time), time + dt

    def cond(carry):
        _, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
