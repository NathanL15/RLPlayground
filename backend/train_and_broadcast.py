from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import torch
import websockets
from stable_baselines3 import PPO

from rl_env import RocketLeague1v1Env


DEFAULT_ELO = 1000.0
MIN_ESTIMATED_TRAIN_ELO = 800.0
TRAIN_DEVICE = os.getenv("RL_DEVICE", "auto")
TRAIN_ONLY_MODE = os.getenv("RL_TRAIN_ONLY", "0") == "1"


def configure_torch_runtime(logger: logging.Logger) -> None:
    """Tune Torch CPU threading for local training throughput."""
    cpu_count = os.cpu_count() or 1
    # Keep interop threads conservative to reduce scheduling overhead.
    intra_threads = max(1, cpu_count)
    interop_threads = max(1, min(4, cpu_count // 2))

    try:
        torch.set_num_threads(intra_threads)
        torch.set_num_interop_threads(interop_threads)
        logger.info(
            "torch runtime configured: device=%s intra_threads=%s interop_threads=%s",
            TRAIN_DEVICE,
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
        )
    except Exception as e:
        logger.warning("failed to configure torch threading: %s", e)


def configure_logging() -> tuple[logging.Logger, Path]:
    """Configure console+file logging and return logger and metrics path."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training.log"
    metrics_file = log_dir / "training_metrics.jsonl"

    logger = logging.getLogger("rlplayground")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("logging initialized: log_file=%s metrics_file=%s", log_file, metrics_file)
    return logger, metrics_file


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


def elo_expected(r_a: float, r_b: float) -> float:
    """Compute expected win rate for player A vs player B using Elo formula."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, result_a: float, k: float = 16.0) -> float:
    """
    Update Elo rating for player A after match.
    
    Args:
        r_a: Current Elo of A
        r_b: Current Elo of B
        result_a: Match result (1.0 win, 0.5 draw, 0.0 loss)
        k: K-factor (higher = more volatile)
    
    Returns:
        New Elo for A
    """
    ea = elo_expected(r_a, r_b)
    return r_a + k * (result_a - ea)


def make_model(env: RocketLeague1v1Env, seed: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs={"net_arch": [512, 512, 256]},
        seed=seed,
        device=TRAIN_DEVICE,
    )


class SelfPlayManager:
    def __init__(self, logger: logging.Logger, metrics_file: Path, chunk_timesteps: int = 32768) -> None:
        self.logger = logger
        self.metrics_file = metrics_file
        self.chunk_timesteps = chunk_timesteps
        # Save latest checkpoint every iteration to ensure safe resume after any stop.
        self.checkpoint_interval_minutes = int(os.getenv("RL_CHECKPOINT_INTERVAL_MINUTES", "20"))
        self.snapshot_every = int(os.getenv("RL_SNAPSHOT_EVERY", "5"))
        self.eval_every = int(os.getenv("RL_EVAL_EVERY", "10"))
        self.eval_matches = int(os.getenv("RL_EVAL_MATCHES", "5"))
        self.eval_opponents = int(os.getenv("RL_EVAL_OPPONENTS", "3"))
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pool_snapshots_dir = self.models_dir / "pool_snapshots"
        self.pool_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_info_path = self.models_dir / "checkpoint_info.json"
        self.trainer_state_path = self.models_dir / "trainer_state.json"
        self.best_avg_elo = float("-inf")
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.iteration = 0
        self.phase = "chase"
        self.phase_step_total = 0
        self.pool_max_size = 20
        self.next_pool_snapshot_id = 1
        # opponent_pool stores: (params1, params2, iteration, elo_rating, snapshot_id)
        self.opponent_pool: List[Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = []
        
        # Track match results for Elo updates: (opponent_idx, win_rate_from_train_bot1, num_games)
        self.opponent_match_stats: Dict[int, Tuple[float, int]] = {}  # idx -> (win_sum, num_games)

        # Training envs (single-agent wrappers with opponent callbacks).
        self.env_train_bot1 = RocketLeague1v1Env(controlled_bot=1)
        self.env_train_bot2 = RocketLeague1v1Env(controlled_bot=2)

        # Live models are used by opponent callbacks and by the broadcast arena.
        self.model1_live = make_model(RocketLeague1v1Env(controlled_bot=1), seed=101)
        self.model2_live = make_model(RocketLeague1v1Env(controlled_bot=2), seed=202)

        # Trainable models.
        self.model1_train = make_model(self.env_train_bot1, seed=11)
        self.model2_train = make_model(self.env_train_bot2, seed=22)

        # Resume from last checkpoint when available.
        self._load_latest_checkpoint_if_available()

        # Sync live from train at startup.
        self._sync_live_models()

        # Opponent policies for training envs.
        self.env_train_bot1.set_opponent_policy(self._opponent_action_from_live_bot2)
        self.env_train_bot2.set_opponent_policy(self._opponent_action_from_live_bot1)
        
        # Current opponent pool index (for stats tracking).
        self.current_opponent_idx = None

    def _pool_snapshot_paths(self, snapshot_id: int) -> tuple[Path, Path]:
        return (
            self.pool_snapshots_dir / f"pool_{snapshot_id}_bot1_policy.zip",
            self.pool_snapshots_dir / f"pool_{snapshot_id}_bot2_policy.zip",
        )

    def _save_pool_snapshot(self, snapshot_id: int) -> None:
        bot1_path, bot2_path = self._pool_snapshot_paths(snapshot_id)
        self.model1_train.save(str(bot1_path))
        self.model2_train.save(str(bot2_path))

    def _delete_pool_snapshot(self, snapshot_id: int) -> None:
        bot1_path, bot2_path = self._pool_snapshot_paths(snapshot_id)
        for path in (bot1_path, bot2_path):
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning("failed deleting pool snapshot %s: %s", path.name, e)

    def _estimate_current_training_elo(self) -> float:
        if not self.opponent_pool:
            return DEFAULT_ELO

        avg_opp_elo = sum(elo for _, _, _, elo, _ in self.opponent_pool) / len(self.opponent_pool)
        return max(MIN_ESTIMATED_TRAIN_ELO, avg_opp_elo - 100.0)

    def _load_trainer_state_if_available(self) -> None:
        if not self.trainer_state_path.exists():
            return

        try:
            state = json.loads(self.trainer_state_path.read_text(encoding="utf-8"))
            self.iteration = int(state.get("iteration", 0))
            self.phase = str(state.get("phase", "chase"))
            self.best_avg_elo = float(state.get("best_avg_elo", float("-inf")))
            self.next_pool_snapshot_id = int(state.get("next_pool_snapshot_id", 1))
            loaded_pool: List[Tuple[Dict[str, Any], Dict[str, Any], int, float, int]] = []

            for pool_entry in state.get("opponent_pool", []):
                snapshot_id = int(pool_entry["snapshot_id"])
                pool_iteration = int(pool_entry["iteration"])
                pool_elo = float(pool_entry.get("elo", DEFAULT_ELO))
                bot1_path, bot2_path = self._pool_snapshot_paths(snapshot_id)

                if not bot1_path.exists() or not bot2_path.exists():
                    self.logger.warning(
                        "skipping pool snapshot %s because saved policy files are missing",
                        snapshot_id,
                    )
                    continue

                loaded_bot1 = PPO.load(str(bot1_path), env=self.env_train_bot1, device=TRAIN_DEVICE)
                loaded_bot2 = PPO.load(str(bot2_path), env=self.env_train_bot2, device=TRAIN_DEVICE)
                loaded_pool.append(
                    (
                        cast(Any, loaded_bot1.get_parameters()),
                        cast(Any, loaded_bot2.get_parameters()),
                        pool_iteration,
                        pool_elo,
                        snapshot_id,
                    )
                )

            self.opponent_pool = loaded_pool
            self.opponent_match_stats = {idx: (0.0, 0) for idx in range(len(self.opponent_pool))}
            self.logger.info(
                "restored trainer state: iteration=%s phase=%s pool_size=%s best_avg_elo=%.1f",
                self.iteration,
                self.phase,
                len(self.opponent_pool),
                self.best_avg_elo,
            )
        except Exception as e:
            self.logger.exception("failed loading trainer state, continuing with weights only: %s", e)

    def _save_trainer_state(self) -> None:
        state = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": self.iteration,
            "phase": self.phase,
            "best_avg_elo": self.best_avg_elo,
            "next_pool_snapshot_id": self.next_pool_snapshot_id,
            "neutral_elo": DEFAULT_ELO,
            "opponent_pool": [
                {
                    "iteration": pool_iteration,
                    "elo": pool_elo,
                    "snapshot_id": snapshot_id,
                }
                for _, _, pool_iteration, pool_elo, snapshot_id in self.opponent_pool
            ],
        }
        self.trainer_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_latest_checkpoint_if_available(self) -> None:
        latest_bot1 = self.models_dir / "latest_bot1_policy.zip"
        latest_bot2 = self.models_dir / "latest_bot2_policy.zip"

        if not latest_bot1.exists() or not latest_bot2.exists():
            self.logger.info("no latest checkpoints found; training starts fresh")
            return

        try:
            loaded_bot1 = PPO.load(str(latest_bot1), env=self.env_train_bot1, device=TRAIN_DEVICE)
            loaded_bot2 = PPO.load(str(latest_bot2), env=self.env_train_bot2, device=TRAIN_DEVICE)
            self.model1_train.set_parameters(cast(Any, loaded_bot1.get_parameters()), exact_match=True)
            self.model2_train.set_parameters(cast(Any, loaded_bot2.get_parameters()), exact_match=True)
            self._load_trainer_state_if_available()
            self.logger.info("loaded latest checkpoints: %s and %s", latest_bot1.name, latest_bot2.name)
        except Exception as e:
            self.logger.exception("failed loading latest checkpoints, starting fresh: %s", e)

    def _save_checkpoint_pair(self, prefix: str) -> None:
        """Save both trainable models using a clear shared prefix."""
        bot1_path = self.models_dir / f"{prefix}_bot1_policy.zip"
        bot2_path = self.models_dir / f"{prefix}_bot2_policy.zip"
        self.model1_train.save(str(bot1_path))
        self.model2_train.save(str(bot2_path))

    def _write_checkpoint_info(self, avg_elo: float, is_best: bool) -> None:
        info = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": self.iteration,
            "avg_elo": avg_elo,
            "best_avg_elo": self.best_avg_elo,
            "neutral_elo": DEFAULT_ELO,
            "phase": self.phase,
            "pool_size": len(self.opponent_pool),
            "latest_checkpoint_interval_minutes": self.checkpoint_interval_minutes,
            "is_best_update": is_best,
            "latest": {
                "bot1": "latest_bot1_policy.zip",
                "bot2": "latest_bot2_policy.zip",
            },
            "best": {
                "bot1": "best_bot1_policy.zip",
                "bot2": "best_bot2_policy.zip",
            },
            "trainer_state": "trainer_state.json",
        }
        self.checkpoint_info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def _maybe_save_checkpoints(self, avg_elo: float) -> None:
        # Always save "latest" every iteration so stopping/restarting resumes from the most recent weights.
        now_utc = datetime.now(timezone.utc)
        should_save_latest = True
        is_best = avg_elo > self.best_avg_elo

        if should_save_latest:
            self._save_checkpoint_pair("latest")
            # Keep a record of the intended checkpoint cadence for reference.
            self.next_latest_checkpoint_at = now_utc + timedelta(minutes=self.checkpoint_interval_minutes)

        if is_best:
            self.best_avg_elo = avg_elo
            self._save_checkpoint_pair("best")

        if should_save_latest or is_best:
            self._save_trainer_state()
            self._write_checkpoint_info(avg_elo=avg_elo, is_best=is_best)
            self.logger.info(
                "checkpoint saved: latest=%s best=%s iter=%s avg_elo=%.1f best_avg_elo=%.1f",
                should_save_latest,
                is_best,
                self.iteration,
                avg_elo,
                self.best_avg_elo,
            )

    def save_final_checkpoint(self) -> None:
        """Always save a last-on-exit checkpoint pair so overnight runs are recoverable."""
        try:
            elo_values = [elo for _, _, _, elo, _ in self.opponent_pool]
            avg_elo = float(np.mean(elo_values)) if elo_values else DEFAULT_ELO
            self._save_checkpoint_pair("latest")
            self._save_trainer_state()
            self._write_checkpoint_info(avg_elo=avg_elo, is_best=False)
            self.logger.info("final checkpoint saved: iter=%s avg_elo=%.1f", self.iteration, avg_elo)
        except Exception as e:
            self.logger.exception("failed to save final checkpoint: %s", e)

    def _append_metrics_row(self, row: Dict[str, Any]) -> None:
        """Append one JSON line to the metrics file for offline analysis."""
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _idle_policy(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        return np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32)

    def _weak_policy(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        obs = env.get_observation(bot_id)
        # Observation is frame-stacked, so decode from latest frame.
        frame_dim = env.base_obs_dim
        latest = obs[-frame_dim:]
        rel_ball_x = float(latest[20])
        rel_ball_y = float(latest[21])
        throttle = np.clip(0.65 + 0.35 * np.sign(rel_ball_x), -1.0, 1.0)
        steering = np.clip(rel_ball_y * 1.8, -1.0, 1.0)
        jump = 1.0 if (abs(rel_ball_y) < 0.15 and rel_ball_x > 0.2 and random.random() < 0.05) else 0.0
        boost = 1.0 if rel_ball_x > 0.35 and random.random() < 0.3 else 0.0
        return np.array([throttle, steering, jump, boost], dtype=np.float32)

    def _pfsp_sample_opponent(self) -> int | None:
        """
        Sample opponent using Prioritized Fictitious Self-Play (PFSP).
        
        Prefers opponents at ~50% win-rate against current training bot.
        If no matches recorded yet, sample uniformly.
        
        Returns:
            Index in opponent_pool, or None if pool is empty.
        """
        if not self.opponent_pool:
            return None
        
        current_elo = self._estimate_current_training_elo()
        
        # Compute PFSP score for each opponent.
        scores = []
        for _, _, _, opp_elo, _ in self.opponent_pool:
            # Expected win rate if we played this opponent.
            expected_win = elo_expected(current_elo, opp_elo)
            
            # PFSP score: prefer opponents close to 50% win rate.
            # Score closer to 0.5 gets higher weight.
            pfsp_score = 1.0 - abs(expected_win - 0.5) / 0.5  # Range [0, 1]
            pfsp_score = pfsp_score ** 2  # Sharpen preferences
            scores.append(pfsp_score)
        
        # Normalize and sample.
        total_score = sum(scores)
        if total_score < 1e-8:
            return random.randint(0, len(self.opponent_pool) - 1)
        
        probabilities = [s / total_score for s in scores]
        return np.random.choice(len(self.opponent_pool), p=probabilities)

    def _update_training_phase(self) -> None:
        # Approximate curriculum using trained experience budget.
        trained_steps = (self.iteration + 1) * self.chunk_timesteps
        if trained_steps < 250_000:
            self.phase = "chase"
            self.env_train_bot1.set_training_phase("chase")
            self.env_train_bot2.set_training_phase("chase")
            self.env_train_bot1.set_opponent_policy(self._idle_policy)
            self.env_train_bot2.set_opponent_policy(self._idle_policy)
        elif trained_steps < 550_000:
            self.phase = "shoot"
            self.env_train_bot1.set_training_phase("shoot")
            self.env_train_bot2.set_training_phase("shoot")
            self.env_train_bot1.set_opponent_policy(self._weak_policy)
            self.env_train_bot2.set_opponent_policy(self._weak_policy)
        elif trained_steps < 1_000_000:
            self.phase = "weak"
            self.env_train_bot1.set_training_phase("weak")
            self.env_train_bot2.set_training_phase("weak")
            self.env_train_bot1.set_opponent_policy(self._weak_policy)
            self.env_train_bot2.set_opponent_policy(self._weak_policy)
        else:
            self.phase = "full"
            self.env_train_bot1.set_training_phase("full")
            self.env_train_bot2.set_training_phase("full")
            self.env_train_bot1.set_opponent_policy(self._opponent_action_from_live_bot2)
            self.env_train_bot2.set_opponent_policy(self._opponent_action_from_live_bot1)

        if self.phase in ("chase", "shoot"):
            ent = 0.02
        elif self.phase == "weak":
            ent = 0.012
        else:
            ent = 0.005
        self.model1_train.ent_coef = ent
        self.model2_train.ent_coef = ent

    def _sync_live_models(self) -> None:
        """
        Sync live models using PFSP sampling from the opponent pool during full self-play phase.
        Early phases use fixed weak opponents for stability.
        """
        with self.lock:
            # During early phases, use fixed weak opponent.
            if self.phase in ("chase", "shoot", "weak"):
                self.model1_live.set_parameters(cast(Any, self.model1_train.get_parameters()), exact_match=True)
                self.model2_live.set_parameters(cast(Any, self.model2_train.get_parameters()), exact_match=True)
                self.current_opponent_idx = None
                return
            
            # In full phase, use PFSP-sampled opponent.
            opp_idx = self._pfsp_sample_opponent()
            if opp_idx is None:
                # Pool empty, use current training bot.
                self.model1_live.set_parameters(cast(Any, self.model1_train.get_parameters()), exact_match=True)
                self.model2_live.set_parameters(cast(Any, self.model2_train.get_parameters()), exact_match=True)
                self.current_opponent_idx = None
            else:
                params1, params2, _, _, _ = self.opponent_pool[opp_idx]
                self.model1_live.set_parameters(cast(Any, params1), exact_match=True)
                self.model2_live.set_parameters(cast(Any, params2), exact_match=True)
                self.current_opponent_idx = opp_idx

    def _snapshot_pool(self) -> None:
        """Snapshot current models into pool with initial Elo rating."""
        if self.snapshot_every <= 0 or self.iteration % self.snapshot_every != 0:
            return

        params1 = copy.deepcopy(self.model1_train.get_parameters())
        params2 = copy.deepcopy(self.model2_train.get_parameters())
        
        snapshot_id = self.next_pool_snapshot_id
        self.next_pool_snapshot_id += 1

        # Initialize new snapshot at a lower neutral Elo so early ratings are less inflated.
        new_elo = DEFAULT_ELO

        self._save_pool_snapshot(snapshot_id)
        self.opponent_pool.append((params1, params2, self.iteration, new_elo, snapshot_id))
        self.opponent_match_stats[len(self.opponent_pool) - 1] = (0.0, 0)  # (win_sum, num_games)
        self.logger.info(
            "snapshot added: iteration=%s pool_size=%s snapshot_elo=%.1f snapshot_id=%s",
            self.iteration,
            len(self.opponent_pool),
            new_elo,
            snapshot_id,
        )
        
        # Prune pool if over capacity, removing worst performers (lowest Elo).
        if len(self.opponent_pool) > self.pool_max_size:
            # Find index of lowest Elo opponent.
            min_elo_idx = min(range(len(self.opponent_pool)), key=lambda i: self.opponent_pool[i][3])
            removed_elo = self.opponent_pool[min_elo_idx][3]
            removed_snapshot_id = self.opponent_pool[min_elo_idx][4]
            self.opponent_pool.pop(min_elo_idx)
            self._delete_pool_snapshot(removed_snapshot_id)
            if min_elo_idx in self.opponent_match_stats:
                del self.opponent_match_stats[min_elo_idx]
            # Re-index stats (this is a simplification; ideally track by ID).
            self.logger.info(
                "snapshot pruned: removed_index=%s removed_elo=%.1f removed_snapshot_id=%s pool_size=%s",
                min_elo_idx,
                removed_elo,
                removed_snapshot_id,
                len(self.opponent_pool),
            )

    def _eval_opponent_strength(self) -> None:
        """
        Periodically evaluate training models against pool opponents to update Elo ratings.
        Simplified: sample a few matches to estimate win rates.
        """
        if self.eval_every <= 0 or self.iteration % self.eval_every != 0 or self.phase != "full" or len(self.opponent_pool) < 2:
            return
        
        # Run a few eval matches against a sample of opponents.
        num_eval_matches = min(max(1, self.eval_opponents), len(self.opponent_pool))
        eval_indices = random.sample(range(len(self.opponent_pool)), num_eval_matches)
        
        for opp_idx in eval_indices:
            params1, params2, _, opp_elo, _ = self.opponent_pool[opp_idx]
            
            # Create eval environment and set opponent.
            eval_env = RocketLeague1v1Env(controlled_bot=1, max_steps=2200)
            eval_env.reset()
            
            # Create a temporary model for opponent.
            temp_opponent_model = make_model(eval_env, seed=9999)
            temp_opponent_model.set_parameters(cast(Any, params2), exact_match=True)
            
            # Play 5 quick matches.
            train_wins = 0
            matches_per_opp = max(1, self.eval_matches)
            for _ in range(matches_per_opp):
                obs, _ = eval_env.reset()
                done = False
                train_touched_last = False
                
                while not done:
                    # Training bot 1 action.
                    action1, _ = self.model1_train.predict(obs, deterministic=False)
                    
                    # Opponent bot 2 action.
                    obs2 = eval_env.get_observation(2)
                    action2, _ = temp_opponent_model.predict(obs2, deterministic=False)
                    
                    obs, reward, terminated, truncated, info = eval_env.step(action1)
                    
                    if info.get("touched_ball"):
                        train_touched_last = True
                    
                    done = terminated or truncated
                    
                    if terminated:
                        scored_by = info.get("scored_by", 0)
                        if scored_by == 1:
                            train_wins += 1
                        elif scored_by == 2 and train_touched_last:
                            # Own goal counts as loss.
                            pass
            
            # Estimate win rate from eval matches.
            eval_win_rate = train_wins / float(matches_per_opp)
            
            # Update Elo for opponent.
            train_elo = self._estimate_current_training_elo()
            new_opp_elo = elo_update(opp_elo, train_elo, 1.0 - eval_win_rate, k=8.0)
            
            # Update pool.
            p1, p2, it, _, snapshot_id = self.opponent_pool[opp_idx]
            self.opponent_pool[opp_idx] = (p1, p2, it, new_opp_elo, snapshot_id)
            self.logger.info(
                "elo update: opp_idx=%s eval_win_rate=%.3f old_elo=%.1f new_elo=%.1f",
                opp_idx,
                eval_win_rate,
                opp_elo,
                new_opp_elo,
            )

    def _opponent_action_from_live_bot1(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        obs = env.get_observation(bot_id)
        with self.lock:
            action, _ = self.model1_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    def _opponent_action_from_live_bot2(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        obs = env.get_observation(bot_id)
        with self.lock:
            action, _ = self.model2_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    def predict_live_action(self, bot_id: int, obs: np.ndarray) -> np.ndarray:
        with self.lock:
            if bot_id == 1:
                action, _ = self.model1_live.predict(obs, deterministic=False)
            else:
                action, _ = self.model2_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    def training_loop(self) -> None:
        while not self.stop_event.is_set():
            iteration_started = datetime.now(timezone.utc).isoformat()
            try:
                self._update_training_phase()
                self.model1_train.learn(total_timesteps=self.chunk_timesteps, reset_num_timesteps=False)
                self.model2_train.learn(total_timesteps=self.chunk_timesteps, reset_num_timesteps=False)
                self._snapshot_pool()
                self._eval_opponent_strength()
                self._sync_live_models()
                self.iteration += 1

                elo_values = [elo for _, _, _, elo, _ in self.opponent_pool]
                avg_elo = float(np.mean(elo_values)) if elo_values else DEFAULT_ELO
                min_elo = float(np.min(elo_values)) if elo_values else DEFAULT_ELO
                max_elo = float(np.max(elo_values)) if elo_values else DEFAULT_ELO

                metric_row = {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "iteration": self.iteration,
                    "phase": self.phase,
                    "chunk_timesteps": self.chunk_timesteps,
                    "trained_steps_total": self.iteration * self.chunk_timesteps,
                    "pool_size": len(self.opponent_pool),
                    "current_opponent_idx": self.current_opponent_idx,
                    "avg_elo": avg_elo,
                    "min_elo": min_elo,
                    "max_elo": max_elo,
                    "ent_coef": float(self.model1_train.ent_coef),
                    "iteration_started_utc": iteration_started,
                }
                self._append_metrics_row(metric_row)
                self._maybe_save_checkpoints(avg_elo=avg_elo)

                self.logger.info(
                    "trainer iteration=%s phase=%s pool=%s opp_idx=%s avg_elo=%.1f elo_range=[%.1f, %.1f] ent=%.4f steps=%s",
                    self.iteration,
                    self.phase,
                    len(self.opponent_pool),
                    self.current_opponent_idx,
                    avg_elo,
                    min_elo,
                    max_elo,
                    float(self.model1_train.ent_coef),
                    self.iteration * self.chunk_timesteps,
                )
            except Exception as e:
                self.logger.exception("training loop error at iteration=%s: %s", self.iteration, e)
                error_row = {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "iteration": self.iteration,
                    "phase": self.phase,
                    "error": str(e),
                }
                self._append_metrics_row(error_row)
                self.stop_event.set()


class BroadcastServer:
    def __init__(self, manager: SelfPlayManager, logger: logging.Logger, host: str = "localhost", port: int = 8001) -> None:
        self.manager = manager
        self.logger = logger
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.env_watch = RocketLeague1v1Env(controlled_bot=1, max_steps=10_000_000)
        self.env_watch.reset()

    async def ws_handler(self, websocket: websockets.WebSocketServerProtocol):
        self.clients.add(websocket)
        try:
            await websocket.send(json.dumps({"type": "connected"}))
            async for _ in websocket:
                # No client messages are required, but we keep the socket open.
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    async def _broadcast_state(self, payload: dict) -> None:
        if not self.clients:
            return

        message = json.dumps(payload)
        dead = []
        for ws in self.clients:
            try:
                await ws.send(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.clients.discard(ws)

    async def simulation_loop(self, hz: float = 30.0) -> None:
        dt = 1.0 / hz
        while True:
            obs1 = self.env_watch.get_observation(1)
            obs2 = self.env_watch.get_observation(2)

            action1 = self.manager.predict_live_action(1, obs1)
            action2 = self.manager.predict_live_action(2, obs2)

            _, scored_by, _ = self.env_watch.simulate_step(action1, action2)
            if scored_by != 0:
                self.env_watch._reset_kickoff()

            payload = self.env_watch.get_broadcast_state()
            payload["training_iteration"] = self.manager.iteration
            payload["scored_by"] = scored_by

            await self._broadcast_state(payload)
            await asyncio.sleep(dt)

    async def run(self) -> None:
        async with websockets.serve(self.ws_handler, self.host, self.port):
            self.logger.info("[ws] listening on ws://%s:%s", self.host, self.port)
            await self.simulation_loop(hz=30.0)


def main() -> None:
    logger, metrics_file = configure_logging()
    configure_torch_runtime(logger)
    chunk_timesteps = int(os.getenv("RL_CHUNK_TIMESTEPS", "32768"))
    manager = SelfPlayManager(logger=logger, metrics_file=metrics_file, chunk_timesteps=chunk_timesteps)

    logger.info(
        "runtime config: train_only=%s chunk_timesteps=%s snapshot_every=%s eval_every=%s eval_matches=%s eval_opponents=%s",
        TRAIN_ONLY_MODE,
        manager.chunk_timesteps,
        manager.snapshot_every,
        manager.eval_every,
        manager.eval_matches,
        manager.eval_opponents,
    )

    if TRAIN_ONLY_MODE:
        try:
            manager.training_loop()
        except KeyboardInterrupt:
            logger.info("shutting down train-only mode via keyboard interrupt")
        finally:
            manager.stop_event.set()
            manager.save_final_checkpoint()
            logger.info("shutdown complete")
        return

    trainer_thread = threading.Thread(target=manager.training_loop, daemon=True)
    trainer_thread.start()

    server = BroadcastServer(manager=manager, logger=logger)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("shutting down via keyboard interrupt")
    except Exception as e:
        logger.exception("fatal server error: %s", e)
    finally:
        manager.stop_event.set()
        trainer_thread.join(timeout=2.0)
        manager.save_final_checkpoint()
        logger.info("shutdown complete")


if __name__ == "__main__":
    main()
