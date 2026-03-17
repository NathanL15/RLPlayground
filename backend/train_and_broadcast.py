from __future__ import annotations

import asyncio
import json
import threading
from typing import Set

import numpy as np
import websockets
from stable_baselines3 import PPO

from rl_env import RocketLeague1v1Env


def make_model(env: RocketLeague1v1Env, seed: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"net_arch": [256, 256, 128]},
        seed=seed,
        device="cpu",
    )


class SelfPlayManager:
    def __init__(self, chunk_timesteps: int = 8192) -> None:
        self.chunk_timesteps = chunk_timesteps
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.iteration = 0

        # Training envs (single-agent wrappers with opponent callbacks).
        self.env_train_bot1 = RocketLeague1v1Env(controlled_bot=1)
        self.env_train_bot2 = RocketLeague1v1Env(controlled_bot=2)

        # Live models are used by opponent callbacks and by the broadcast arena.
        self.model1_live = make_model(RocketLeague1v1Env(controlled_bot=1), seed=101)
        self.model2_live = make_model(RocketLeague1v1Env(controlled_bot=2), seed=202)

        # Trainable models.
        self.model1_train = make_model(self.env_train_bot1, seed=11)
        self.model2_train = make_model(self.env_train_bot2, seed=22)

        # Sync live from train at startup.
        self._sync_live_models()

        # Opponent policies for training envs.
        self.env_train_bot1.set_opponent_policy(self._opponent_action_from_live_bot2)
        self.env_train_bot2.set_opponent_policy(self._opponent_action_from_live_bot1)

    def _sync_live_models(self) -> None:
        with self.lock:
            self.model1_live.set_parameters(self.model1_train.get_parameters(), exact_match=True)
            self.model2_live.set_parameters(self.model2_train.get_parameters(), exact_match=True)

    def _opponent_action_from_live_bot1(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        obs = env.get_observation(bot_id)
        with self.lock:
            action, _ = self.model1_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.int64)

    def _opponent_action_from_live_bot2(self, env: RocketLeague1v1Env, bot_id: int) -> np.ndarray:
        obs = env.get_observation(bot_id)
        with self.lock:
            action, _ = self.model2_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.int64)

    def predict_live_action(self, bot_id: int, obs: np.ndarray) -> np.ndarray:
        with self.lock:
            if bot_id == 1:
                action, _ = self.model1_live.predict(obs, deterministic=False)
            else:
                action, _ = self.model2_live.predict(obs, deterministic=False)
        return np.asarray(action, dtype=np.int64)

    def training_loop(self) -> None:
        while not self.stop_event.is_set():
            self.model1_train.learn(total_timesteps=self.chunk_timesteps, reset_num_timesteps=False)
            self.model2_train.learn(total_timesteps=self.chunk_timesteps, reset_num_timesteps=False)
            self._sync_live_models()
            self.iteration += 1
            print(f"[trainer] completed iteration {self.iteration}")


class BroadcastServer:
    def __init__(self, manager: SelfPlayManager, host: str = "localhost", port: int = 8001) -> None:
        self.manager = manager
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
            print(f"[ws] listening on ws://{self.host}:{self.port}")
            await self.simulation_loop(hz=30.0)


def main() -> None:
    manager = SelfPlayManager(chunk_timesteps=8192)
    trainer_thread = threading.Thread(target=manager.training_loop, daemon=True)
    trainer_thread.start()

    server = BroadcastServer(manager=manager)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        manager.stop_event.set()
        trainer_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
