from __future__ import annotations

from collections import deque
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


@dataclass
class BoostPad:
    pos: np.ndarray
    is_big: bool
    active: bool = True
    timer: float = 0.0


BIG_PAD_POSITIONS = [
    (-14.4, -8.0), (-14.4, 8.0),
    (14.4, -8.0), (14.4, 8.0),
    (0.0, -10.8), (0.0, 10.8),
]

SMALL_PAD_POSITIONS = [
    (-3.6, 0.0), (3.6, 0.0),
    (0.0, -4.0), (0.0, 4.0),
    (-7.2, -4.8), (-7.2, 4.8),
    (7.2, -4.8), (7.2, 4.8),
    (-10.8, -4.8), (-10.8, 4.8),
    (10.8, -4.8), (10.8, 4.8),
    (-14.4, 0.0), (14.4, 0.0),
    (-7.2, 0.0), (7.2, 0.0),
    (-3.6, -7.2), (-3.6, 7.2),
    (3.6, -7.2), (3.6, 7.2),
]


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
        max_steps: int = 2200,
        dt: float = 1.0 / 30.0,
        frame_stack: int = 4,
    ) -> None:
        super().__init__()

        if controlled_bot not in (1, 2):
            raise ValueError("controlled_bot must be 1 or 2")

        self.controlled_bot = controlled_bot
        self.opponent_policy = opponent_policy
        self.max_steps = max_steps
        self.dt = dt
        self.gamma = 0.99
        self.frame_stack = max(1, int(frame_stack))
        self.training_phase = "full"

        # Arena dimensions (x length, y width, z height)
        self.half_length = 18.0
        self.half_width = 12.0
        self.ceiling = 8.0

        # Goal zones centered on x back walls (8 units wide out of 24 = large target)
        self.goal_half_width = 4.0
        self.goal_height = 3.2

        # Bot dimensions (axis-aligned box around center)
        self.bot_half_extents = np.array([0.9, 0.65, 0.45], dtype=np.float32)

        # Ball
        self.ball_radius = 0.45

        # Physics tuning (closer to Rocket League values)
        self.gravity = -13.0
        self.ball_restitution = 0.6
        self.wall_friction = 0.998
        self.bot_ground_friction = 0.94
        self.bot_air_drag = 0.997
        self.drive_accel = 16.0
        self.boost_accel = 33.0
        self.max_bot_speed = 14.6
        self.max_bot_speed_boost = 23.0
        self.max_ball_speed = 45.0
        self.turn_rate = 2.8
        self.jump_impulse = 6.5
        self.jump_cooldown_time = 0.8

        # Boost system
        self.max_boost = 100.0
        self.boost_start = 33.0
        self.boost_consumption_rate = 33.3
        self.big_pad_amount = 100.0
        self.small_pad_amount = 12.0
        self.big_pad_respawn_time = 10.0
        self.small_pad_respawn_time = 4.0
        self.big_pad_pickup_radius = 2.0
        self.small_pad_pickup_radius = 1.2

        # Lightweight "vision" rays in bot-local yaw space (radians).
        self.vision_angles = np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2], dtype=np.float32)
        self.vision_max_dist = 24.0

        # Action: [throttle, steering, jump, boost]
        # throttle and steering are continuous in [-1, 1], jump/boost are thresholded from [0, 1].
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation: base feature vector stacked over time for temporal awareness.
        self.base_obs_dim = 69  # Updated: added intercept time features, ball danger, and longer-horizon ball prediction
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_obs_dim * self.frame_stack,),
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
        self.steps_since_touch = {1: 0.0, 2: 0.0}

        self.bot_boost = {1: self.boost_start, 2: self.boost_start}
        self.boost_pads: list[BoostPad] = []
        self.obs_stack: Dict[int, deque[np.ndarray]] = {}
        self.reward_abs_ema = 1.0

        # Potential-based reward shaping: store prev potential per bot for computing PBRS.
        self.prev_potential = {1: 0.0, 2: 0.0}
        
        # PBRS hyperparameters (potential weights).
        self.pbrs_weights = {
            "ball_approach": 0.5,      # Distance to ball (lower is better)
            "goal_progress": 1.0,      # Ball distance to opponent goal (lower is better)
            "defensive": 0.3,          # Ball distance to own goal (higher is better, so inverted)
        }

        self.np_random = np.random.default_rng()
        self.reset()

    def set_opponent_policy(self, policy: Optional[Callable[["RocketLeague1v1Env", int], np.ndarray]]) -> None:
        self.opponent_policy = policy

    def set_training_phase(self, phase: str) -> None:
        allowed = {"chase", "shoot", "weak", "full"}
        self.training_phase = phase if phase in allowed else "full"

    def _normalize_reward(self, raw_reward: float) -> float:
        self.reward_abs_ema = 0.995 * self.reward_abs_ema + 0.005 * abs(raw_reward)
        scaled = raw_reward / max(0.1, self.reward_abs_ema)
        return float(np.clip(scaled, -10.0, 10.0))

    def _compute_potential(self, bot_id: int) -> float:
        """
        Compute potential function for PBRS (potential-based reward shaping).
        
        Potential combines three safe, policy-invariant terms:
        - Phi_ball: negative distance to ball (closer is better)
        - Phi_goal: negative ball distance to opponent goal (closer to goal is better)
        - Phi_def: negative ball distance to own goal (farther from own goal is better)
        
        Returns a scalar potential that is bounded (~-5 to +5 in typical states).
        """
        bot = self.bot1 if bot_id == 1 else self.bot2
        opp_goal = self._opponent_goal_center(bot_id)
        own_goal = self._own_goal_center(bot_id)
        
        # Distance to ball (normalized by arena scale).
        d_car_ball = float(np.linalg.norm(bot.pos - self.ball_pos))
        phi_ball = -self.pbrs_weights["ball_approach"] * (d_car_ball / self.vision_max_dist)
        
        # Ball distance to opponent goal (normalized).
        d_ball_opp_goal = float(np.linalg.norm(self.ball_pos[:2] - opp_goal[:2]))
        phi_goal = -self.pbrs_weights["goal_progress"] * (d_ball_opp_goal / (2.0 * self.half_length))
        
        # Ball distance to own goal (inverted: farther is better).
        # Use 1/(1+d) to keep bounded and smooth.
        d_ball_own_goal = float(np.linalg.norm(self.ball_pos[:2] - own_goal[:2]))
        phi_def = -self.pbrs_weights["defensive"] * (d_ball_own_goal / (2.0 * self.half_length))
        
        potential = phi_ball + phi_goal + phi_def
        return float(np.clip(potential, -5.0, 5.0))

    def _estimate_intercept_time(self, bot: BotState, ball_pos: np.ndarray, ball_vel: np.ndarray) -> float:
        """
        Estimate time (in seconds) for a bot to reach the ball using forward prediction.
        
        Simplification: assume bot travels in a straight line from current position toward
        ball's current position at max speed, and ball follows linear trajectory.
        Returns time or a large value if unreachable.
        """
        rel_pos = ball_pos[:2] - bot.pos[:2]
        dist = float(np.linalg.norm(rel_pos))
        
        # Use average of max speed and current speed for a rough estimate.
        avg_speed = (self.max_bot_speed + float(np.linalg.norm(bot.vel[:2]))) / 2.0 + 0.5
        
        # Assume bot can reach the ball; add small margin for turning/acceleration.
        time_to_reach = max(0.0, (dist / avg_speed) - 0.1)
        return float(np.clip(time_to_reach, 0.0, 3.0))

    def _compute_ball_danger(self) -> float:
        """
        Estimate how close the predicted ball trajectory gets to the defending team's goal.
        
        Returns a value in [0, 1] where 1 = ball is in immediate danger zone.
        Simplified: project ball forward 1 second and check minimum distance to goal.
        """
        # Current bot 1's own goal is at x = -half_length.
        own_goal_x = -self.half_length
        
        # Predict ball position 1 second ahead (clamped to max speed).
        pred_ball = self.ball_pos[:2] + self.ball_vel[:2] * 1.0
        ball_dist_to_goal = abs(float(pred_ball[0]) - own_goal_x)
        
        # Normalize: 0 when far from goal (> 10), 1 when very close (< 2).
        danger = max(0.0, 1.0 - (ball_dist_to_goal / 10.0))
        return float(np.clip(danger, 0.0, 1.0))

    def _rollout_ball_trajectory(self, horizon_steps: int = 60) -> list[np.ndarray]:
        """
        Simulate ball-only physics forward in time and return trajectory.
        
        Args:
            horizon_steps: Number of physics steps to simulate (~60 = 2 seconds at 30 Hz)
            
        Returns:
            List of ball positions [x, y, z] at each step.
        """
        trajectory = [self.ball_pos.copy()]
        ball_pos = self.ball_pos.copy()
        ball_vel = self.ball_vel.copy()
        
        for _ in range(horizon_steps):
            # Apply gravity
            ball_vel[2] += self.gravity * self.dt
            
            # Integrate
            ball_pos += ball_vel * self.dt
            
            # Resolve ball-world collisions (simplified: bounces off walls/floor/ceiling)
            # Floor
            if ball_pos[2] < self.ball_radius:
                ball_pos[2] = self.ball_radius
                if ball_vel[2] < 0:
                    ball_vel[2] *= -self.ball_restitution
            
            # Ceiling
            elif ball_pos[2] > self.ceiling - self.ball_radius:
                ball_pos[2] = self.ceiling - self.ball_radius
                if ball_vel[2] > 0:
                    ball_vel[2] *= -self.ball_restitution
            
            # Side walls
            if ball_pos[1] < -self.half_width + self.ball_radius:
                ball_pos[1] = -self.half_width + self.ball_radius
                if ball_vel[1] < 0:
                    ball_vel[1] *= -self.ball_restitution
            elif ball_pos[1] > self.half_width - self.ball_radius:
                ball_pos[1] = self.half_width - self.ball_radius
                if ball_vel[1] > 0:
                    ball_vel[1] *= -self.ball_restitution
            
            # Back walls (x-axis) - but don't count as goals, just bounce
            if ball_pos[0] < -self.half_length + self.ball_radius:
                ball_pos[0] = -self.half_length + self.ball_radius
                if ball_vel[0] < 0:
                    ball_vel[0] *= -self.ball_restitution
            elif ball_pos[0] > self.half_length - self.ball_radius:
                ball_pos[0] = self.half_length - self.ball_radius
                if ball_vel[0] > 0:
                    ball_vel[0] *= -self.ball_restitution
            
            # Apply wall friction damping
            ball_vel *= self.wall_friction
            
            # Cap max speed
            speed = float(np.linalg.norm(ball_vel))
            if speed > self.max_ball_speed:
                ball_vel *= self.max_ball_speed / speed
            
            trajectory.append(ball_pos.copy())
        
        return trajectory

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
        self.steps_since_touch = {1: 0.0, 2: 0.0}
        
        # Initialize potentials at kickoff.
        self.prev_potential[1] = self._compute_potential(1)
        self.prev_potential[2] = self._compute_potential(2)

        self.bot_boost = {1: self.boost_start, 2: self.boost_start}
        self.boost_pads = []
        for x, y in BIG_PAD_POSITIONS:
            self.boost_pads.append(BoostPad(pos=np.array([x, y], dtype=np.float32), is_big=True))
        for x, y in SMALL_PAD_POSITIONS:
            self.boost_pads.append(BoostPad(pos=np.array([x, y], dtype=np.float32), is_big=False))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.reward_abs_ema = 1.0
        self._reset_kickoff()
        self._prime_obs_stacks()

        obs = self.get_observation(self.controlled_bot)
        info = {"score": dict(self.score)}
        return obs, info

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, int, int]:
        action = np.asarray(action, dtype=np.float32)
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))
        jump = 1 if float(action[2]) > 0.5 else 0
        boost = 1 if len(action) > 3 and float(action[3]) > 0.5 else 0
        return throttle, steering, jump, boost

    def _apply_bot_controls(self, bot: BotState, bot_id: int, action: np.ndarray) -> None:
        throttle, steering, jump, boost_input = self._decode_action(action)

        wheel_contact = self._wheel_contact_ratio(bot)
        traction = 0.25 + 0.75 * wheel_contact

        bot.ang_vel = steering * self.turn_rate * traction
        bot.yaw += bot.ang_vel * self.dt

        forward = np.array([np.cos(bot.yaw), np.sin(bot.yaw), 0.0], dtype=np.float32)

        # Boost logic
        using_boost = boost_input == 1 and self.bot_boost[bot_id] > 0.0
        if using_boost:
            self.bot_boost[bot_id] = max(0.0, self.bot_boost[bot_id] - self.boost_consumption_rate * self.dt)
            accel = self.boost_accel
            current_max_speed = self.max_bot_speed_boost
        else:
            accel = self.drive_accel
            current_max_speed = self.max_bot_speed

        bot.vel[:2] += (forward[:2] * (throttle * accel * traction * self.dt)).astype(np.float32)

        speed_xy = float(np.linalg.norm(bot.vel[:2]))
        if speed_xy > current_max_speed:
            bot.vel[:2] = bot.vel[:2] * (current_max_speed / speed_xy)

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

    def _compute_dense_reward(self, bot_id: int, touched: bool, boost_collected: float = 0.0) -> float:
        """
        Compute reward using potential-based reward shaping (PBRS) for safety against
        reward hacking, combined with event-based bonuses (touch quality, boost).
        
        PBRS ensures that the shaped reward preserves optimal policy while densifying learning.
        """
        bot = self.bot1 if bot_id == 1 else self.bot2
        
        # Compute potential-based shaping.
        curr_potential = self._compute_potential(bot_id)
        pbrs_reward = self.gamma * curr_potential - self.prev_potential[bot_id]
        
        # Track for next step.
        self.prev_potential[bot_id] = curr_potential
        
        reward = pbrs_reward
        
        # Update touch tracking for situation awareness.
        self.steps_since_touch[bot_id] = 0.0 if touched else self.steps_since_touch[bot_id] + self.dt
        
        # Touch quality reward (event-based): reward good touches, penalize bad ones.
        # This is independent of potential shaping since it's conditioned on the action taken.
        if touched:
            opp_goal = self._opponent_goal_center(bot_id)
            to_goal = opp_goal[:2] - self.ball_pos[:2]
            to_goal_norm = float(np.linalg.norm(to_goal)) + 1e-8
            to_goal_unit = to_goal / to_goal_norm
            
            ball_speed_xy = float(np.linalg.norm(self.ball_vel[:2])) + 1e-8
            ball_dir = self.ball_vel[:2] / ball_speed_xy
            hit_quality = float(np.dot(ball_dir, to_goal_unit))
            
            # Reduced magnitudes compared to before to prevent reward hacking.
            touch_reward = 0.4 + 0.6 * max(0.0, hit_quality)
            if hit_quality < -0.2:
                touch_reward -= 0.3
            reward += touch_reward
        
        # Boost collection reward (event-based).
        if boost_collected > 0:
            reward += 0.15 * (boost_collected / self.max_boost)
        
        # Jump penalty: small cost for unnecessary jumps.
        if self.jump_used_step[bot_id] > 0:
            ball_above = self.ball_pos[2] > 1.5
            ball_close = float(np.linalg.norm(bot.pos - self.ball_pos)) < 4.0
            if not (ball_above and ball_close):
                reward -= 0.05  # Reduced from 0.12
        
        # Anti-idle penalty (very small to avoid dominating PBRS).
        self_speed = float(np.linalg.norm(bot.vel[:2]))
        current_dist = float(np.linalg.norm(bot.pos - self.ball_pos))
        if self_speed < 0.3 and current_dist > 4.0:
            reward -= 0.001
        
        # Curriculum shaping in early training phases.
        if self.training_phase == "chase":
            # In chase phase, add small bonus for touching.
            if touched:
                reward += 0.2
        elif self.training_phase == "shoot":
            # In shoot phase, reward touches that move ball toward goal (already in PBRS).
            if touched:
                reward += 0.1
        
        # Per-step baseline penalty to encourage efficiency.
        reward -= 0.0005
        
        return reward

    def _update_boost_pads(self) -> Dict[int, float]:
        """Update pad timers and check for bot pickups. Returns boost collected per bot."""
        collected = {1: 0.0, 2: 0.0}
        for pad in self.boost_pads:
            if not pad.active:
                pad.timer -= self.dt
                if pad.timer <= 0.0:
                    pad.active = True
                    pad.timer = 0.0
                continue

            pickup_radius = self.big_pad_pickup_radius if pad.is_big else self.small_pad_pickup_radius
            amount = self.big_pad_amount if pad.is_big else self.small_pad_amount
            respawn = self.big_pad_respawn_time if pad.is_big else self.small_pad_respawn_time

            for bot_id, bot in [(1, self.bot1), (2, self.bot2)]:
                dist = float(np.linalg.norm(bot.pos[:2] - pad.pos))
                if dist < pickup_radius and self.bot_boost[bot_id] < self.max_boost:
                    if pad.is_big:
                        self.bot_boost[bot_id] = self.max_boost
                    else:
                        self.bot_boost[bot_id] = min(self.max_boost, self.bot_boost[bot_id] + amount)
                    collected[bot_id] += amount
                    pad.active = False
                    pad.timer = respawn
                    break
        return collected

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

        boost_collected = self._update_boost_pads()

        rewards = {
            1: self._compute_dense_reward(1, touched[1], boost_collected[1]),
            2: self._compute_dense_reward(2, touched[2], boost_collected[2]),
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
                self.np_random.uniform(-1.0, 1.0),
                self.np_random.uniform(-1.0, 1.0),
                float(self.np_random.integers(0, 2)),
                float(self.np_random.integers(0, 2)),
            ],
            dtype=np.float32,
        )

    def _get_opponent_action(self, opponent_bot_id: int) -> np.ndarray:
        if self.opponent_policy is None:
            return self._sample_random_action()

        try:
            action = self.opponent_policy(self, opponent_bot_id)
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (4,):
                return self._sample_random_action()
            return action
        except Exception:
            return self._sample_random_action()

    def step(self, action: np.ndarray):
        self.steps += 1

        if self.controlled_bot == 1:
            action_bot1 = np.asarray(action, dtype=np.float32)
            action_bot2 = self._get_opponent_action(2)
        else:
            action_bot1 = self._get_opponent_action(1)
            action_bot2 = np.asarray(action, dtype=np.float32)

        rewards, scored_by, touched = self.simulate_step(action_bot1, action_bot2)

        terminated = scored_by != 0
        truncated = self.steps >= self.max_steps
        reward = self._normalize_reward(rewards[self.controlled_bot])
        info = {
            "scored_by": scored_by,
            "touched_ball": touched[self.controlled_bot],
            "score": dict(self.score),
        }

        if terminated:
            self._reset_kickoff()
            self._prime_obs_stacks()
        else:
            self._append_obs_stacks()

        obs = self.get_observation(self.controlled_bot)

        return obs, float(reward), terminated, truncated, info

    def _compute_nearest_boost_features(self, bot: BotState) -> np.ndarray:
        nearest_active = None
        nearest_dist = 1e9
        for pad in self.boost_pads:
            if not pad.active:
                continue
            d = float(np.linalg.norm(bot.pos[:2] - pad.pos))
            if d < nearest_dist:
                nearest_dist = d
                nearest_active = pad

        if nearest_active is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        rel = nearest_active.pos - bot.pos[:2]
        rel_local = self._world_to_local_xy(bot.yaw, rel)
        rel_local_norm = rel_local / np.array([self.half_length, self.half_width], dtype=np.float32)
        dist_norm = float(np.clip(nearest_dist / self.vision_max_dist, 0.0, 1.0))
        return np.array([rel_local_norm[0], rel_local_norm[1], dist_norm], dtype=np.float32)

    def _velocity_heading_alignment(self, bot: BotState) -> float:
        speed = float(np.linalg.norm(bot.vel[:2])) + 1e-8
        vel_dir = bot.vel[:2] / speed
        forward = np.array([np.cos(bot.yaw), np.sin(bot.yaw)], dtype=np.float32)
        return float(np.dot(vel_dir, forward))

    def _build_base_observation(self, perspective_bot: int) -> np.ndarray:
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
        
        # Compute longer-horizon prediction using analytic rollout (more accurate for longer times).
        traj = self._rollout_ball_trajectory(horizon_steps=60)  # 60 steps = 2 seconds at 30 Hz
        if len(traj) > 30:  # Get position at ~1 second
            pred_ball_long = traj[30]
            pred_ball_long_local = self._world_to_local_xy(self_bot.yaw, pred_ball_long[:2] - self_bot.pos[:2])
        else:
            pred_ball_long_local = np.zeros(2, dtype=np.float32)

        wall_ray = np.array([self._ray_wall_distance_norm(self_bot, float(a)) for a in self.vision_angles], dtype=np.float32)
        ball_ray = np.array([self._ray_ball_alignment(self_bot, float(a)) for a in self.vision_angles], dtype=np.float32)
        rel_ball_vel_local = self._world_to_local_xy(self_bot.yaw, self.ball_vel[:2] - self_bot.vel[:2])
        nearest_boost = self._compute_nearest_boost_features(self_bot)
        on_ground = 1.0 if self._wheel_contact_ratio(self_bot) > 0.5 else 0.0
        ball_z_local = (float(self.ball_pos[2]) - float(self_bot.pos[2])) / self.ceiling
        heading_alignment = self._velocity_heading_alignment(self_bot)
        ball_dist_to_opp_goal = float(np.linalg.norm(self.ball_pos[:2] - opp_goal[:2])) / (2.0 * self.half_length)
        ball_dist_to_own_goal = float(np.linalg.norm(self.ball_pos[:2] - own_goal[:2])) / (2.0 * self.half_length)
        touch_age = min(1.0, self.steps_since_touch[perspective_bot] / 5.0)
        
        # Situation awareness features (strategic signals to guide behavior).
        intercept_time_self = self._estimate_intercept_time(self_bot, self.ball_pos, self.ball_vel)
        intercept_time_opp = self._estimate_intercept_time(opp_bot, self.ball_pos, self.ball_vel)
        intercept_time_delta = (intercept_time_opp - intercept_time_self) / 3.0  # Normalize to ~[-1, 1]
        ball_danger = self._compute_ball_danger()

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
                np.array([self.bot_boost[perspective_bot] / self.max_boost], dtype=np.float32),
                np.array([self.bot_boost[3 - perspective_bot] / self.max_boost], dtype=np.float32),
                rel_ball_vel_local / np.array([self.max_ball_speed, self.max_ball_speed], dtype=np.float32),
                nearest_boost,
                np.array([on_ground], dtype=np.float32),
                np.array([ball_z_local], dtype=np.float32),
                np.array([heading_alignment], dtype=np.float32),
                np.array([ball_dist_to_opp_goal], dtype=np.float32),
                np.array([ball_dist_to_own_goal], dtype=np.float32),
                np.array([touch_age], dtype=np.float32),
                np.array([intercept_time_self / 3.0], dtype=np.float32),  # Normalize to [0, 1]
                np.array([intercept_time_opp / 3.0], dtype=np.float32),   # Normalize to [0, 1]
                np.array([intercept_time_delta], dtype=np.float32),       # Delta: ~[-1, 1]
                np.array([ball_danger], dtype=np.float32),
                pred_ball_long_local / np.array([self.half_length, self.half_width], dtype=np.float32),  # 2-sec prediction
                wall_ray,
                ball_ray,
            ]
        ).astype(np.float32)

        return obs

    def _prime_obs_stacks(self) -> None:
        self.obs_stack = {
            1: deque(maxlen=self.frame_stack),
            2: deque(maxlen=self.frame_stack),
        }
        base1 = self._build_base_observation(1)
        base2 = self._build_base_observation(2)
        for _ in range(self.frame_stack):
            self.obs_stack[1].append(base1.copy())
            self.obs_stack[2].append(base2.copy())

    def _append_obs_stacks(self) -> None:
        self.obs_stack[1].append(self._build_base_observation(1))
        self.obs_stack[2].append(self._build_base_observation(2))

    def get_observation(self, perspective_bot: int) -> np.ndarray:
        if perspective_bot not in self.obs_stack or len(self.obs_stack[perspective_bot]) == 0:
            self._prime_obs_stacks()
        return np.concatenate(list(self.obs_stack[perspective_bot]), axis=0).astype(np.float32)

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

    def get_broadcast_state(self) -> dict:
        return {
            "ball": [float(self.ball_pos[0]), float(self.ball_pos[1]), float(self.ball_pos[2])],
            "bot1": [float(self.bot1.pos[0]), float(self.bot1.pos[1]), float(self.bot1.pos[2]), float(self.bot1.yaw)],
            "bot2": [float(self.bot2.pos[0]), float(self.bot2.pos[1]), float(self.bot2.pos[2]), float(self.bot2.yaw)],
            "score": {"bot1": int(self.score[1]), "bot2": int(self.score[2])},
            "bot1_boost": float(self.bot_boost[1]),
            "bot2_boost": float(self.bot_boost[2]),
            "boost_pads": [pad.active for pad in self.boost_pads],
        }
