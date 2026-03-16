"""Lightning module with manual optimization for MARL Discrete SAC.

Lightning owns:
  - checkpointing
  - logging
  - evaluation hooks
  - multi-optimizer stepping
  - experiment reproducibility

Lightning does NOT own:
  - the rollout loop (that lives in ``rl.rollout``)
  - SUMO simulation (that lives in ``env.sumo_env``)
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from tensordict import TensorDict
from torch import Tensor

from models.marl_discrete_sac import MARLDiscreteSAC
from rl.losses import DiscreteSACLossComputer, SACLossOutput
from rl.optimizers import make_optimizer
from rl.replay import make_replay_buffer
from rl.rollout import RolloutWorker
from marl_env.sumo_env import TrafficSignalEnv


class TrafficMARLModule(L.LightningModule):
    """Lightning training shell for graph-based MARL Discrete SAC.

    Parameters
    ----------
    env_cfg : dict
        Kwargs for :class:`TrafficSignalEnv`.
    model_cfg : dict
        Kwargs for :class:`MARLDiscreteSAC` (obs_dim, num_actions, encoder_cfg, etc.).
    train_cfg : dict
        Training hyper-parameters (lr, gamma, batch_size, replay_capacity, etc.).
    """

    def __init__(
        self,
        env_cfg: dict,
        model_cfg: dict,
        train_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Manual optimization — we control backward and optimizer steps
        self.automatic_optimization = False

        train_cfg = train_cfg or {}
        default_opt = {
            "name": "adam",
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "meta": {"hyper_lr": 1e-7, "min_lr": 1e-6, "max_lr": 1e-2},
        }
        self.optimizer_actor_cfg = train_cfg.get("optimizer_actor", train_cfg.get("optimizer", default_opt))
        self.optimizer_critic_cfg = train_cfg.get("optimizer_critic", train_cfg.get("optimizer", default_opt))
        self.optimizer_alpha_cfg = train_cfg.get("optimizer_alpha", train_cfg.get("optimizer", default_opt))
        self.gamma = train_cfg.get("gamma", 0.99)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.replay_capacity = train_cfg.get("replay_capacity", 100_000)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.updates_per_step = train_cfg.get("updates_per_step", 1)
        self.target_update_freq = train_cfg.get("target_update_freq", 1)
        self.eval_interval = train_cfg.get("eval_interval", 10)
        self.steps_per_epoch = train_cfg.get("steps_per_epoch", 100)

        # Built in setup()
        self.agent: MARLDiscreteSAC | None = None
        self.loss_fn: DiscreteSACLossComputer | None = None
        self.replay_buffer: Any = None
        self.env: TrafficSignalEnv | None = None
        self.rollout_worker: RolloutWorker | None = None

        self._env_cfg = env_cfg
        self._model_cfg = model_cfg
        self._global_env_step = 0

    # ==================================================================
    # Lightning hooks
    # ==================================================================
    def setup(self, stage: str | None = None) -> None:
        """Initialize environment, agent, replay buffer, rollout worker."""
        # --- Environment ---
        self.env = TrafficSignalEnv(**self._env_cfg)
        td = self.env.reset()

        obs_dim = td.get("graph_observation", td["agents", "observation"]).shape[-1]
        num_actions = self.env.num_actions

        # --- Agent ---
        self.agent = MARLDiscreteSAC(
            obs_dim=obs_dim,
            num_actions=num_actions,
            **self._model_cfg,
        ).to(self.device)

        # --- Loss computer ---
        self.loss_fn = DiscreteSACLossComputer(self.agent, gamma=self.gamma)

        # --- Replay buffer ---
        self.replay_buffer = make_replay_buffer(
            capacity=self.replay_capacity,
            batch_size=self.batch_size,
        )

        # --- Rollout worker ---
        self.rollout_worker = RolloutWorker(
            env=self.env, agent=self.agent, device=self.device,
        )

        self.env.close()

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Three separate optimizers: actor, critic, alpha."""
        assert self.agent is not None
        opt_actor = make_optimizer(
            list(self.agent.encoder.parameters())
            + list(self.agent.actor.parameters()),
            self.optimizer_actor_cfg,
        )
        opt_critic = make_optimizer(self.agent.critic.parameters(), self.optimizer_critic_cfg)
        opt_alpha = make_optimizer([self.agent.log_alpha], self.optimizer_alpha_cfg)
        return [opt_actor, opt_critic, opt_alpha]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> None:
        """One training iteration: collect → replay → update.

        Since we use manual optimization and a custom rollout loop,
        the ``batch`` argument from the dataloader is ignored.
        We use a dummy dataloader that just yields step indices.
        """
        assert self.agent is not None
        assert self.loss_fn is not None
        assert self.rollout_worker is not None
        assert self.replay_buffer is not None

        opt_actor, opt_critic, opt_alpha = self.optimizers()  # type: ignore[misc]

        # --- Collect environment steps ---
        transitions = self.rollout_worker.collect_steps(
            n_steps=self.steps_per_epoch
        )
        for t in transitions:
            self.replay_buffer.add(t)
        self._global_env_step += len(transitions)

        # --- Skip update if not enough data ---
        if self._global_env_step < self.warmup_steps:
            return

        # --- Gradient updates ---
        for _ in range(self.updates_per_step):
            sample = self.replay_buffer.sample()
            sample = sample.to(self.device)

            loss_out: SACLossOutput = self.loss_fn(sample)

            # Critic update
            opt_critic.zero_grad()
            self.manual_backward(loss_out.critic_loss)
            opt_critic.step()

            # Actor update
            opt_actor.zero_grad()
            self.manual_backward(loss_out.actor_loss)
            opt_actor.step()

            # Alpha update
            opt_alpha.zero_grad()
            self.manual_backward(loss_out.alpha_loss)
            opt_alpha.step()

            # Target update
            if self.global_step % self.target_update_freq == 0:
                self.agent.soft_update_target()

        # --- Logging ---
        self.log("train/critic_loss", loss_out.critic_loss, prog_bar=True)
        self.log("train/actor_loss", loss_out.actor_loss, prog_bar=True)
        self.log("train/alpha_loss", loss_out.alpha_loss)
        self.log("train/alpha", loss_out.alpha)
        self.log("train/q1_mean", loss_out.q1_mean)
        self.log("train/q2_mean", loss_out.q2_mean)
        self.log("train/entropy", loss_out.entropy)
        self.log("train/env_steps", float(self._global_env_step))

    # ------------------------------------------------------------------
    # Validation / Evaluation
    # ------------------------------------------------------------------
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Run one evaluation episode and log traffic metrics."""
        assert self.rollout_worker is not None
        self.agent.eval()
        _, info = self.rollout_worker.collect_episode(deterministic=True)
        self.agent.train()

        self.log("val/episode_return", info["episode_return"], prog_bar=True)
        self.log("val/episode_length", info["episode_length"])

    def on_train_end(self) -> None:
        if self.env is not None:
            self.env.close()
