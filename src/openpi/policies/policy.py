from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
        # Built lazily by `infer_rtc`, one compiled graph per (schedule, use_vjp).
        self._rtc_samplers: dict[tuple[str, bool], Any] = {}

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _rtc_sampler(self, prefix_attention_schedule: str, use_vjp: bool):
        """The jitted RTC sampler for one (schedule, use_vjp) pair, built once.

        Those two select which computation runs rather than parameterizing it,
        so they are static and each pair is its own compiled graph. Everything a
        control loop varies per cycle -- delay, horizon, guidance ceiling, noise,
        step count -- stays traced, so a running robot never recompiles.
        """
        key = (prefix_attention_schedule, use_vjp)
        if key not in self._rtc_samplers:
            self._rtc_samplers[key] = nnx_utils.module_jit(
                self._model.sample_actions_rtc,
                static_argnames=("prefix_attention_schedule", "use_vjp"),
            )
        return self._rtc_samplers[key]

    def infer_rtc(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        inference_delay: int = 0,
        execution_horizon: int | None = None,
        max_guidance_weight: float = 10.0,
        prefix_attention_schedule: str = "exp",
        use_vjp: bool = True,
        num_steps: int | None = None,
    ) -> dict:
        """Infer with Real-Time Chunking guidance, returning the normalized chunk too.

        `infer` above unnormalizes before returning, but RTC prefixes are in
        normalized model space: a caller that fed back the unnormalized chunk
        would guide in the wrong frame with no error anywhere. So both are
        returned, and `actions` still means exactly what it means in `infer`.

        `prev_chunk_left_over` of None takes the unguided `sample_actions` path,
        which is a separate compiled graph and the cheaper one; guidance costs a
        vector-Jacobian product through the denoiser at every step.

        Args:
            obs: Observation dict, as for `infer`.
            noise: [ah, ad] or [b, ah, ad] initial noise, at the model's own
                action dim; drawn from the policy rng when omitted.
            prev_chunk_left_over: [ah, ad] previous chunk's unexecuted tail in
                normalized action space, index 0 aligned with the new chunk's
                index 0. None disables guidance.
            inference_delay: Leading timesteps already committed to the robot.
            execution_horizon: Where prefix influence decays to zero; clamped to
                the tail's own length by the sampler.
            max_guidance_weight: Guidance ceiling.
            prefix_attention_schedule: Soft-mask shape, one of the
                `rtc.PrefixAttentionSchedule` literals.
            use_vjp: True for PI's pseudo-inverse correction, False for the
                first-order form LeRobot's port computes.
            num_steps: Flow-matching steps; the policy's own when omitted.

        Returns:
            The `infer` dict plus `normalized_actions`, the model-space chunk.
        """
        if self._is_pytorch_model:
            raise NotImplementedError("infer_rtc is implemented for the JAX path only")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps
        if noise is not None:
            noise = jnp.asarray(noise)
            sample_kwargs["noise"] = noise[None, ...] if noise.ndim == 2 else noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if prev_chunk_left_over is None:
            chunk = self._sample_actions(sample_rng, observation, **sample_kwargs)
        else:
            prev = jnp.asarray(prev_chunk_left_over, dtype=jnp.float32)
            chunk = self._rtc_sampler(prefix_attention_schedule, use_vjp)(
                sample_rng,
                observation,
                prev_chunk_left_over=prev[None, ...] if prev.ndim == 2 else prev,
                inference_delay=inference_delay,
                execution_horizon=(
                    prev.shape[-2] if execution_horizon is None else execution_horizon
                ),
                max_guidance_weight=max_guidance_weight,
                prefix_attention_schedule=prefix_attention_schedule,
                use_vjp=use_vjp,
                **sample_kwargs,
            )
        model_time = time.monotonic() - start_time

        normalized = np.asarray(chunk[0, ...])
        outputs = self._output_transform(
            {"state": np.asarray(inputs["state"][0, ...]), "actions": normalized.copy()}
        )
        outputs["normalized_actions"] = normalized
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
