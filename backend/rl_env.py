from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class BotState:
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    ang_vel: float


class RocketLeague1v1Env(gym.Env):
    """
    Lightweight 1v1 Rocket League style environment with custom NumPy physics.

    This is a single-agent Gymnasium env where one bot is controlled by the policy,
    and the other bot is controlled by an opponent policy callback.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        controlled_bot: int = 1,
        opponent_policy: Optional[Callable[["RocketLeague1v1Env", int], np.ndarray]] = None,
        max_steps: int = 1500,
        dt: float = 1.0 / 30.0,
    ) -> None:
        super().__init__()

        if controlled_bot not in (1, 2):
            raise ValueError("controlled_bot must be 1 or 2")

        self.controlled_bot = controlled_bot
        self.opponent_policy = opponent_policy
        self.max_steps = max_steps
        self.dt = dt

        # Arena dimensions (x length, y width, z height)
        self.half_length = 18.0
        self.half_width = 12.0
        self.ceiling = 8.0
        self.corner_radius = 3.6

        # Goal zones centered on x back walls (8 units wide out of 24 = large target)
        self.goal_half_width = 4.0
        self.goal_height = 3.2

        # Bot dimensions (axis-aligned box around center)
        self.bot_half_extents = np.array([0.9, 0.65, 0.45], dtype=np.float32)

        # Ball
        self.ball_radius = 0.45

        # Physics tuning
        self.gravity = -18.0
        self.ball_restitution = 0.86
        self.wall_friction = 0.995
        self.bot_ground_friction = 0.92
        self.bot_air_drag = 0.995
        self.drive_accel = 24.0
        self.max_bot_speed = 14.0
        self.max_ball_speed = 32.0
        self.turn_rate = 3.8
        self.jump_impulse = 7.8
        self.jump_cooldown_time = 0.45

        # Lightweight "vision" rays in bot-local yaw space (radians).
        self.vision_angles = np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2], dtype=np.float32)
        self.vision_max_dist = 24.0

        # Action: [throttle, steering, jump]
        # throttle: {-1,0,1}, steering: {-1,0,1}, jump: {0,1}
        self.action_space = spaces.MultiDiscrete([3, 3, 2])

        # Observation combines normalized state + relational features + simple vision rays.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(50,),
            dtype=np.float32,
        )

        self.bot1: BotState
        self.bot2: BotState
        self.ball_pos: np.ndarray
        self.ball_vel: np.ndarray

        self.steps = 0
        self.last_ball_dist = {1: 0.0, 2: 0.0}
        self.score = {1: 0, 2: 0}
        self.last_touched_bot = 0  # 0=none, 1=bot1, 2=bot2
        self.prev_jump_pressed = {1: 0, 2: 0}
        self.jump_cooldown = {1: 0.0, 2: 0.0}
        self.jump_used_step = {1: 0.0, 2: 0.0}

        self.np_random = np.random.default_rng()
        self.reset()

    def set_opponent_policy(self, policy: Optional[Callable[["RocketLeague1v1Env", int], np.ndarray]]) -> None:
        self.opponent_policy = policy

    def _spawn_bot(self, bot_id: int) -> BotState:
        if bot_id == 1:
            pos = np.array([-8.0, 0.0, self.bot_half_extents[2]], dtype=np.float32)
            yaw = 0.0
        else:
            pos = np.array([8.0, 0.0, self.bot_half_extents[2]], dtype=np.float32)
            yaw = np.pi
        return BotState(pos=pos, vel=np.zeros(3, dtype=np.float32), yaw=yaw, ang_vel=0.0)

    def _reset_kickoff(self) -> None:
        self.bot1 = self._spawn_bot(1)
        self.bot2 = self._spawn_bot(2)

        self.ball_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        self.ball_vel = np.array(
            [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.2, 0.2), 0.0],
            dtype=np.float32,
        )

        self.last_ball_dist[1] = float(np.linalg.norm(self.bot1.pos - self.ball_pos))
        self.last_ball_dist[2] = float(np.linalg.norm(self.bot2.pos - self.ball_pos))
        self.last_touched_bot = 0
        self.prev_jump_pressed = {1: 0, 2: 0}
        self.jump_cooldown = {1: 0.0, 2: 0.0}
        self.jump_used_step = {1: 0.0, 2: 0.0}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self._reset_kickoff()

        obs = self.get_observation(self.controlled_bot)
        info = {"score": dict(self.score)}
        return obs, info

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, int]:
        action = np.asarray(action, dtype=np.int64)
        throttle = float(action[0] - 1)
        steering = float(action[1] - 1)
        jump = int(action[2])
        return throttle, steering, jump

    def _apply_bot_controls(self, bot: BotState, bot_id: int, action: np.ndarray) -> None:
        throttle, steering, jump = self._decode_action(action)

        wheel_contact = self._wheel_contact_ratio(bot)
        traction = 0.25 + 0.75 * wheel_contact

        bot.ang_vel = steering * self.turn_rate * traction
        bot.yaw += bot.ang_vel * self.dt

        forward = np.array([np.cos(bot.yaw), np.sin(bot.yaw), 0.0], dtype=np.float32)

        bot.vel[:2] += (forward[:2] * (throttle * self.drive_accel * traction * self.dt)).astype(np.float32)

        speed_xy = float(np.linalg.norm(bot.vel[:2]))
        if speed_xy > self.max_bot_speed:
            bot.vel[:2] = bot.vel[:2] * (self.max_bot_speed / speed_xy)

        on_ground = wheel_contact > 0.5
        self.jump_cooldown[bot_id] = max(0.0, self.jump_cooldown[bot_id] - self.dt)
        jump_rising_edge = jump == 1 and self.prev_jump_pressed[bot_id] == 0
        can_jump = on_ground and self.jump_cooldown[bot_id] <= 0.0 and jump_rising_edge
        if can_jump:
            bot.vel[2] = self.jump_impulse
            self.jump_cooldown[bot_id] = self.jump_cooldown_time
            self.jump_used_step[bot_id] = 1.0
        else:
            self.jump_used_step[bot_id] = 0.0

        self.prev_jump_pressed[bot_id] = jump

        bot.vel[2] += self.gravity * self.dt

        if on_ground:
            # Wheel-like grip and rolling resistance.
            bot.vel[:2] *= self.bot_ground_friction * (0.985 + 0.015 * traction)
        else:
            bot.vel *= self.bot_air_drag

        bot.pos += bot.vel * self.dt

        self._resolve_bot_world_collision(bot)

    def _resolve_bot_world_collision(self, bot: BotState) -> None:
        ext = self.bot_half_extents

        # Floor / ceiling
        if bot.pos[2] - ext[2] < 0.0:
            bot.pos[2] = ext[2]
            if bot.vel[2] < 0:
                bot.vel[2] *= -0.2
        elif bot.pos[2] + ext[2] > self.ceiling:
            bot.pos[2] = self.ceiling - ext[2]
            if bot.vel[2] > 0:
                bot.vel[2] *= -0.2

        # Side walls
        if bot.pos[1] - ext[1] < -self.half_width:
            bot.pos[1] = -self.half_width + ext[1]
            if bot.vel[1] < 0:
                bot.vel[1] *= -0.4
        elif bot.pos[1] + ext[1] > self.half_width:
            bot.pos[1] = self.half_width - ext[1]
            if bot.vel[1] > 0:
                bot.vel[1] *= -0.4

        # Back walls
        if bot.pos[0] - ext[0] < -self.half_length:
            bot.pos[0] = -self.half_length + ext[0]
            if bot.vel[0] < 0:
                bot.vel[0] *= -0.4
        elif bot.pos[0] + ext[0] > self.half_length:
            bot.pos[0] = self.half_length - ext[0]
            if bot.vel[0] > 0:
                bot.vel[0] *= -0.4

        self._resolve_corner_collision_bot(bot)

    def _resolve_ball_world_collision_and_goals(self) -> int:
        scored_by = 0

        # Goal checks on x back walls before applying x bounce.
        if self.ball_pos[0] + self.ball_radius > self.half_length:
            in_goal = (
                abs(self.ball_pos[1]) <= self.goal_half_width
                and self.ball_pos[2] <= self.goal_height
                and self.ball_pos[2] >= 0.0
            )
            if in_goal:
                scored_by = 1
            else:
                self.ball_pos[0] = self.half_length - self.ball_radius
                if self.ball_vel[0] > 0:
                    self.ball_vel[0] *= -self.ball_restitution

        elif self.ball_pos[0] - self.ball_radius < -self.half_length:
            in_goal = (
                abs(self.ball_pos[1]) <= self.goal_half_width
                and self.ball_pos[2] <= self.goal_height
                and self.ball_pos[2] >= 0.0
            )
            if in_goal:
                scored_by = 2
            else:
                self.ball_pos[0] = -self.half_length + self.ball_radius
                if self.ball_vel[0] < 0:
                    self.ball_vel[0] *= -self.ball_restitution

        # Side walls
        if self.ball_pos[1] + self.ball_radius > self.half_width:
            self.ball_pos[1] = self.half_width - self.ball_radius
            if self.ball_vel[1] > 0:
                self.ball_vel[1] *= -self.ball_restitution
        elif self.ball_pos[1] - self.ball_radius < -self.half_width:
            self.ball_pos[1] = -self.half_width + self.ball_radius
            if self.ball_vel[1] < 0:
                self.ball_vel[1] *= -self.ball_restitution

        # Floor / ceiling
        if self.ball_pos[2] - self.ball_radius < 0.0:
            self.ball_pos[2] = self.ball_radius
            if self.ball_vel[2] < 0:
                self.ball_vel[2] *= -self.ball_restitution
        elif self.ball_pos[2] + self.ball_radius > self.ceiling:
            self.ball_pos[2] = self.ceiling - self.ball_radius
            if self.ball_vel[2] > 0:
                self.ball_vel[2] *= -self.ball_restitution

        self._resolve_corner_collision_ball()

        self.ball_vel *= self.wall_friction
        speed = float(np.linalg.norm(self.ball_vel))
        if speed > self.max_ball_speed:
            self.ball_vel *= self.max_ball_speed / speed

        return scored_by

    def _sphere_box_collision(self, bot: BotState, bot_id: int) -> bool:
        bot_min = bot.pos - self.bot_half_extents
        bot_max = bot.pos + self.bot_half_extents

        closest = np.clip(self.ball_pos, bot_min, bot_max)
        delta = self.ball_pos - closest
        dist = float(np.linalg.norm(delta))

        if dist >= self.ball_radius:
            return False
        
        # Track which bot last touched the ball
        self.last_touched_bot = bot_id

        if dist < 1e-8:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            normal = (delta / dist).astype(np.float32)

        penetration = self.ball_radius - dist
        self.ball_pos += normal * penetration

        rel_vel = self.ball_vel - bot.vel
        vn = float(np.dot(rel_vel, normal))
        if vn < 0:
            impulse = -(1.0 + self.ball_restitution) * vn
            self.ball_vel += normal * impulse

        # Extra forward "hit" feel from the bot direction.
        forward = np.array([np.cos(bot.yaw), np.sin(bot.yaw), 0.0], dtype=np.float32)
        self.ball_vel += forward * 2.0

        return True

    def _compute_dense_reward(self, bot_id: int, touched: bool) -> float:
        bot = self.bot1 if bot_id == 1 else self.bot2
        current_dist = float(np.linalg.norm(bot.pos - self.ball_pos))
        delta_dist = self.last_ball_dist[bot_id] - current_dist
        self.last_ball_dist[bot_id] = current_dist

        reward = 0.15 * delta_dist
        if touched:
            reward += 2.0

        # Encourage facing and driving into meaningful play-space direction.
        to_ball = self.ball_pos[:2] - bot.pos[:2]
        to_ball_norm = float(np.linalg.norm(to_ball)) + 1e-8
        to_ball_unit = to_ball / to_ball_norm
        forward = np.array([np.cos(bot.yaw), np.sin(bot.yaw)], dtype=np.float32)
        reward += 0.01 * float(np.dot(forward, to_ball_unit))

        # Encourage sending the ball toward opponent goal.
        opp_goal = self._opponent_goal_center(bot_id)
        to_goal = opp_goal[:2] - self.ball_pos[:2]
        to_goal_norm = float(np.linalg.norm(to_goal)) + 1e-8
        to_goal_unit = to_goal / to_goal_norm
        ball_goal_progress = float(np.dot(self.ball_vel[:2], to_goal_unit)) / self.max_ball_speed
        reward += 0.06 * ball_goal_progress

        # If we touch and send ball toward opponent goal, reward that extra.
        if touched and ball_goal_progress > 0:
            reward += 0.8 * ball_goal_progress

        # Tiny anti-idle penalty when far from the ball.
        self_speed = float(np.linalg.norm(bot.vel[:2]))
        if self_speed < 0.3 and current_dist > 3.0:
            reward -= 0.004

        # Penalize unnecessary jumps to reduce spam behavior.
        reward -= 0.03 * self.jump_used_step[bot_id]

        reward -= 0.001
        return reward

    def simulate_step(self, action_bot1: np.ndarray, action_bot2: np.ndarray) -> Tuple[Dict[int, float], int, Dict[int, bool]]:
        self._apply_bot_controls(self.bot1, 1, action_bot1)
        self._apply_bot_controls(self.bot2, 2, action_bot2)

        # Integrate ball
        self.ball_vel[2] += self.gravity * self.dt
        self.ball_pos += self.ball_vel * self.dt

        touched = {1: False, 2: False}
        if self._sphere_box_collision(self.bot1, 1):
            touched[1] = True
        if self._sphere_box_collision(self.bot2, 2):
            touched[2] = True

        scored_by = self._resolve_ball_world_collision_and_goals()

        rewards = {
            1: self._compute_dense_reward(1, touched[1]),
            2: self._compute_dense_reward(2, touched[2]),
        }

        if scored_by == 1:
            # Positive X goal belongs to bot1's attacking side.
            rewards[1] += 100.0
            rewards[2] -= 100.0
            # If conceding side (bot2) touched last, this is an own-goal by bot2.
            if self.last_touched_bot == 2:
                rewards[1] += 40.0
                rewards[2] -= 40.0
            self.score[1] += 1
        elif scored_by == 2:
            # Negative X goal belongs to bot2's attacking side.
            rewards[2] += 100.0
            rewards[1] -= 100.0
            # If conceding side (bot1) touched last, this is an own-goal by bot1.
            if self.last_touched_bot == 1:
                rewards[2] += 40.0
                rewards[1] -= 40.0
            self.score[2] += 1

        return rewards, scored_by, touched

    def _sample_random_action(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.integers(0, 3),
                self.np_random.integers(0, 3),
                self.np_random.integers(0, 2),
            ],
            dtype=np.int64,
        )

    def _get_opponent_action(self, opponent_bot_id: int) -> np.ndarray:
        if self.opponent_policy is None:
            return self._sample_random_action()

        try:
            action = self.opponent_policy(self, opponent_bot_id)
            action = np.asarray(action, dtype=np.int64)
            if action.shape != (3,):
                return self._sample_random_action()
            return action
        except Exception:
            return self._sample_random_action()

    def step(self, action: np.ndarray):
        self.steps += 1

        if self.controlled_bot == 1:
            action_bot1 = np.asarray(action, dtype=np.int64)
            action_bot2 = self._get_opponent_action(2)
        else:
            action_bot1 = self._get_opponent_action(1)
            action_bot2 = np.asarray(action, dtype=np.int64)

        rewards, scored_by, touched = self.simulate_step(action_bot1, action_bot2)

        terminated = scored_by != 0
        truncated = self.steps >= self.max_steps

        obs = self.get_observation(self.controlled_bot)
        reward = rewards[self.controlled_bot]
        info = {
            "scored_by": scored_by,
            "touched_ball": touched[self.controlled_bot],
            "score": dict(self.score),
        }

        if terminated:
            self._reset_kickoff()

        return obs, float(reward), terminated, truncated, info

    def get_observation(self, perspective_bot: int) -> np.ndarray:
        b1 = self.bot1
        b2 = self.bot2

        if perspective_bot == 1:
            self_bot, opp_bot = b1, b2
        else:
            self_bot, opp_bot = b2, b1

        rel_ball_world = self.ball_pos[:2] - self_bot.pos[:2]
        rel_opp_world = opp_bot.pos[:2] - self_bot.pos[:2]
        rel_ball_local = self._world_to_local_xy(self_bot.yaw, rel_ball_world)
        rel_opp_local = self._world_to_local_xy(self_bot.yaw, rel_opp_world)

        opp_goal = self._opponent_goal_center(perspective_bot)
        own_goal = self._own_goal_center(perspective_bot)
        to_opp_goal_local = self._world_to_local_xy(self_bot.yaw, opp_goal[:2] - self_bot.pos[:2])
        to_own_goal_local = self._world_to_local_xy(self_bot.yaw, own_goal[:2] - self_bot.pos[:2])

        self_speed = float(np.linalg.norm(self_bot.vel[:2]))
        ball_speed = float(np.linalg.norm(self.ball_vel[:2]))
        dist_to_ball = float(np.linalg.norm(rel_ball_world))

        # Predict short-horizon ball location in local coordinates (anticipation feature).
        pred_t1 = 0.35
        pred_t2 = 0.75
        pred_ball1_world = self.ball_pos[:2] + self.ball_vel[:2] * pred_t1
        pred_ball2_world = self.ball_pos[:2] + self.ball_vel[:2] * pred_t2
        pred_ball1_local = self._world_to_local_xy(self_bot.yaw, pred_ball1_world - self_bot.pos[:2])
        pred_ball2_local = self._world_to_local_xy(self_bot.yaw, pred_ball2_world - self_bot.pos[:2])

        wall_ray = np.array([self._ray_wall_distance_norm(self_bot, float(a)) for a in self.vision_angles], dtype=np.float32)
        ball_ray = np.array([self._ray_ball_alignment(self_bot, float(a)) for a in self.vision_angles], dtype=np.float32)

        obs = np.concatenate(
            [
                self_bot.pos / np.array([self.half_length, self.half_width, self.ceiling], dtype=np.float32),
                self_bot.vel / np.array([self.max_bot_speed, self.max_bot_speed, max(1.0, self.jump_impulse)], dtype=np.float32),
                np.array([self_bot.yaw / np.pi], dtype=np.float32),
                opp_bot.pos / np.array([self.half_length, self.half_width, self.ceiling], dtype=np.float32),
                opp_bot.vel / np.array([self.max_bot_speed, self.max_bot_speed, max(1.0, self.jump_impulse)], dtype=np.float32),
                np.array([opp_bot.yaw / np.pi], dtype=np.float32),
                self.ball_pos / np.array([self.half_length, self.half_width, self.ceiling], dtype=np.float32),
                self.ball_vel / np.array([self.max_ball_speed, self.max_ball_speed, self.max_ball_speed], dtype=np.float32),
                rel_ball_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                rel_opp_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                to_opp_goal_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                to_own_goal_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                np.array([dist_to_ball / self.vision_max_dist], dtype=np.float32),
                np.array([self_speed / self.max_bot_speed], dtype=np.float32),
                np.array([ball_speed / self.max_ball_speed], dtype=np.float32),
                np.array([1.0 if self.last_touched_bot == perspective_bot else 0.0], dtype=np.float32),
                pred_ball1_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                pred_ball2_local / np.array([self.half_length, self.half_width], dtype=np.float32),
                wall_ray,
                ball_ray,
            ]
        ).astype(np.float32)

        return obs

    def _opponent_goal_center(self, bot_id: int) -> np.ndarray:
        x = self.half_length if bot_id == 1 else -self.half_length
        return np.array([x, 0.0, self.goal_height * 0.5], dtype=np.float32)

    def _own_goal_center(self, bot_id: int) -> np.ndarray:
        x = -self.half_length if bot_id == 1 else self.half_length
        return np.array([x, 0.0, self.goal_height * 0.5], dtype=np.float32)

    def _world_to_local_xy(self, yaw: float, vec_xy: np.ndarray) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x_local = c * float(vec_xy[0]) + s * float(vec_xy[1])
        y_local = -s * float(vec_xy[0]) + c * float(vec_xy[1])
        return np.array([x_local, y_local], dtype=np.float32)

    def _ray_wall_distance_norm(self, bot: BotState, angle_offset: float) -> float:
        angle = float(bot.yaw + angle_offset)
        d = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        p = bot.pos[:2]

        ts = []

        if abs(float(d[0])) > 1e-6:
            tx1 = (self.half_length - float(p[0])) / float(d[0])
            tx2 = (-self.half_length - float(p[0])) / float(d[0])
            if tx1 > 0:
                ts.append(tx1)
            if tx2 > 0:
                ts.append(tx2)

        if abs(float(d[1])) > 1e-6:
            ty1 = (self.half_width - float(p[1])) / float(d[1])
            ty2 = (-self.half_width - float(p[1])) / float(d[1])
            if ty1 > 0:
                ts.append(ty1)
            if ty2 > 0:
                ts.append(ty2)

        if not ts:
            return 1.0

        dist = min(ts)
        return float(np.clip(dist / self.vision_max_dist, 0.0, 1.0))

    def _ray_ball_alignment(self, bot: BotState, angle_offset: float) -> float:
        rel = self.ball_pos[:2] - bot.pos[:2]
        dist = float(np.linalg.norm(rel)) + 1e-8
        rel_unit = rel / dist

        angle = float(bot.yaw + angle_offset)
        ray = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        directional = max(0.0, float(np.dot(ray, rel_unit)))
        distance_term = float(np.exp(-dist / self.vision_max_dist))
        return float(np.clip(directional * distance_term, 0.0, 1.0))

    def _wheel_contact_ratio(self, bot: BotState) -> float:
        wheel_points = self._wheel_world_points(bot)
        contacts = 0
        for p in wheel_points:
            if p[2] <= 0.03:
                contacts += 1
        return float(contacts) / 4.0

    def _wheel_world_points(self, bot: BotState) -> np.ndarray:
        # Four wheel centers in local bot space, near the chassis bottom.
        local = np.array(
            [
                [self.bot_half_extents[0] * 0.72, self.bot_half_extents[1] * 0.78, -self.bot_half_extents[2] * 0.95],
                [self.bot_half_extents[0] * 0.72, -self.bot_half_extents[1] * 0.78, -self.bot_half_extents[2] * 0.95],
                [-self.bot_half_extents[0] * 0.72, self.bot_half_extents[1] * 0.78, -self.bot_half_extents[2] * 0.95],
                [-self.bot_half_extents[0] * 0.72, -self.bot_half_extents[1] * 0.78, -self.bot_half_extents[2] * 0.95],
            ],
            dtype=np.float32,
        )

        c = float(np.cos(bot.yaw))
        s = float(np.sin(bot.yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        out = np.zeros((4, 3), dtype=np.float32)
        for i in range(4):
            xy = rot @ local[i, :2]
            out[i, 0] = bot.pos[0] + xy[0]
            out[i, 1] = bot.pos[1] + xy[1]
            out[i, 2] = bot.pos[2] + local[i, 2]
        return out

    def _resolve_corner_collision_ball(self) -> None:
        corner_center_x = self.half_length - self.corner_radius
        corner_center_y = self.half_width - self.corner_radius
        effective_radius = max(0.2, self.corner_radius - self.ball_radius)

        if abs(self.ball_pos[0]) > corner_center_x and abs(self.ball_pos[1]) > corner_center_y:
            cx = corner_center_x * np.sign(self.ball_pos[0])
            cy = corner_center_y * np.sign(self.ball_pos[1])
            delta = self.ball_pos[:2] - np.array([cx, cy], dtype=np.float32)
            dist = float(np.linalg.norm(delta)) + 1e-8
            if dist > effective_radius:
                normal = delta / dist
                self.ball_pos[:2] = np.array([cx, cy], dtype=np.float32) + normal * effective_radius
                vn = float(np.dot(self.ball_vel[:2], normal))
                if vn > 0:
                    self.ball_vel[:2] -= (1.0 + self.ball_restitution) * vn * normal

    def _resolve_corner_collision_bot(self, bot: BotState) -> None:
        corner_center_x = self.half_length - self.corner_radius
        corner_center_y = self.half_width - self.corner_radius
        bot_radius = float(np.linalg.norm(self.bot_half_extents[:2])) * 0.85
        effective_radius = max(0.2, self.corner_radius - bot_radius)

        if abs(bot.pos[0]) > corner_center_x and abs(bot.pos[1]) > corner_center_y:
            cx = corner_center_x * np.sign(bot.pos[0])
            cy = corner_center_y * np.sign(bot.pos[1])
            delta = bot.pos[:2] - np.array([cx, cy], dtype=np.float32)
            dist = float(np.linalg.norm(delta)) + 1e-8
            if dist > effective_radius:
                normal = delta / dist
                bot.pos[:2] = np.array([cx, cy], dtype=np.float32) + normal * effective_radius
                vn = float(np.dot(bot.vel[:2], normal))
                if vn > 0:
                    bot.vel[:2] -= 1.2 * vn * normal

    def get_broadcast_state(self) -> dict:
        return {
            "ball": [float(self.ball_pos[0]), float(self.ball_pos[1]), float(self.ball_pos[2])],
            "bot1": [float(self.bot1.pos[0]), float(self.bot1.pos[1]), float(self.bot1.pos[2]), float(self.bot1.yaw)],
            "bot2": [float(self.bot2.pos[0]), float(self.bot2.pos[1]), float(self.bot2.pos[2]), float(self.bot2.yaw)],
            "score": {"bot1": int(self.score[1]), "bot2": int(self.score[2])},
        }
