"""Lightweight PushT environments reused by GPI policies."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gym import Env, spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st


_PositiveYUp = False


def _to_pygame(point: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    if _PositiveYUp:
        return round(point[0]), surface.get_height() - round(point[1])
    return round(point[0]), round(point[1])


def _light_color(color: SpaceDebugColor) -> SpaceDebugColor:
    scaled = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    return SpaceDebugColor(r=scaled[0], g=scaled[1], b=scaled[2], a=scaled[3])


class _DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        super().__init__()
        self.surface = surface

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        center = _to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, fill_color.as_int(), center, round(radius), 0)
        pygame.draw.circle(
            self.surface,
            _light_color(fill_color).as_int(),
            center,
            round(radius - 4),
            0,
        )
        edge = pos + Vec2d(radius, 0).rotated(angle)
        pygame.draw.lines(
            self.surface,
            fill_color.as_int(),
            False,
            [center, _to_pygame(edge, self.surface)],
        )

    def draw_segment(self, a, b, color):
        pygame.draw.aalines(
            self.surface,
            color.as_int(),
            False,
            [_to_pygame(a, self.surface), _to_pygame(b, self.surface)],
        )

    def draw_fat_segment(self, a, b, radius, outline_color, fill_color):
        p1 = _to_pygame(a, self.surface)
        p2 = _to_pygame(b, self.surface)
        thickness = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], thickness)
        if thickness <= 2:
            return
        ortho = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
        if ortho[0] == ortho[1] == 0:
            return
        norm = radius / (ortho[0] * ortho[0] + ortho[1] * ortho[1]) ** 0.5
        ortho[0] = round(ortho[0] * norm)
        ortho[1] = round(ortho[1] * norm)
        points = [
            (p1[0] - ortho[0], p1[1] - ortho[1]),
            (p1[0] + ortho[0], p1[1] + ortho[1]),
            (p2[0] + ortho[0], p2[1] + ortho[1]),
            (p2[0] - ortho[0], p2[1] - ortho[1]),
        ]
        pygame.draw.polygon(self.surface, fill_color.as_int(), points)
        pygame.draw.circle(self.surface, fill_color.as_int(), p1, round(radius))
        pygame.draw.circle(self.surface, fill_color.as_int(), p2, round(radius))

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        pts = [_to_pygame(v, self.surface) for v in verts]
        pts.append(pts[0])
        pygame.draw.polygon(self.surface, _light_color(fill_color).as_int(), pts)
        if radius > 0:
            for i in range(len(verts)):
                self.draw_fat_segment(
                    verts[i],
                    verts[(i + 1) % len(verts)],
                    radius,
                    fill_color,
                    fill_color,
                )

    def draw_dot(self, size, pos, color):
        pygame.draw.circle(
            self.surface, color.as_int(), _to_pygame(pos, self.surface), round(size)
        )


def _pymunk_to_shapely(body: pymunk.Body, shapes: Sequence[pymunk.Shape]) -> sg.Polygon:
    def _body_to_points(shape: pymunk.Shape) -> Sequence[Tuple[float, float]]:
        if isinstance(shape, pymunk.Poly):
            return [body.local_to_world(v) for v in shape.get_vertices()]
        if isinstance(shape, pymunk.Circle):
            center = body.local_to_world((0, 0))
            return sg.Point(center).buffer(shape.radius).exterior.coords
        raise TypeError("Unsupported shape for conversion")

    all_points = []
    for shape in shapes:
        all_points.extend(_body_to_points(shape))
    return sg.MultiPoint(all_points).convex_hull


class PushTEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy: bool = False,
        block_cog: Optional[Tuple[float, float]] = None,
        damping: Optional[float] = None,
        render_action: bool = True,
        render_size: int = 96,
        reset_to_state: Optional[np.ndarray] = None,
    ) -> None:
        self._seed = None
        self.seed()
        self.window_size = 512
        self.render_size = render_size
        self.sim_hz = 100
        self.k_p = 100
        self.k_v = 20
        self.control_hz = self.metadata["video.frames_per_second"]
        self.legacy = legacy
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([512, 512, 512, 512, 2 * np.pi]),
            shape=(5,),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float64)
        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action
        self.reset_to_state = reset_to_state
        self.window = None
        self.clock = None
        self.screen = None
        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None

        # simulate friction
        self.g_eff = 10.0
        self.mu_trans = 0.0 # 0.6             # translational Coulomb-like friction coefficient#
        self.mu_rot_visc = 0.0 # 0.05         # rotational viscous coefficient (per second)
        self.tau_rot_coulomb = 0.0 # 0.02     # ro
        self.mu_trans = 0.5             
        self.mu_rot_visc = 0.1         
        self.tau_rot_coulomb = 0.02     

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        state = self.reset_to_state
        if state is None:
            rng = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rng.randint(50, 450),
                    rng.randint(50, 450),
                    rng.randint(100, 400),
                    rng.randint(100, 400),
                    rng.randn() * 2 * np.pi - np.pi,
                ]
            )
        self._set_state(state)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def reset_kp_kv(self, kp: float, kv: float) -> None:
        self.k_p = kp
        self.k_v = kv

    def step(self, action: np.ndarray):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt
                # apply friction
                # self._apply_planar_friction(self.block, dt)
                self.space.step(dt)
        goal_geom = _pymunk_to_shapely(
            self._get_goal_pose_body(self.goal_pose), self.block.shapes
        )
        block_geom = _pymunk_to_shapely(self.block, self.block.shapes)
        coverage = goal_geom.intersection(block_geom).area / goal_geom.area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, done, info

    def render(self, mode: str):
        return self._render_frame(mode)

    def _get_obs(self) -> np.ndarray:
        agent = tuple(self.agent.position)
        block = tuple(self.block.position)
        return np.array(agent + block + (self.block.angle % (2 * np.pi),))

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        contacts = int(np.ceil(self.n_contact_points / n_steps))
        return {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": contacts,
        }

    def _render_frame(self, mode: str):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        draw_options = _DrawOptions(canvas)
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            goal_points.append(goal_points[0])
            pygame.draw.polygon(canvas, self.goal_color, goal_points)
        self.space.debug_draw(draw_options)
        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action and self.latest_action is not None:
            coord = (np.array(self.latest_action) / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        if self.legacy:
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = []
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)
        
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 4])
        try:
            handler = self.space.add_default_collision_handler()
        except AttributeError:
            handler = self.space.add_collision_handler(0, 0)
        handler.post_solve = self._handle_collision
        self.n_contact_points = 0
        self.max_score = 50 * 100
        self.success_threshold = 0.95

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        # shape.friction = 1.0         # pusher–block grip
        # shape.elasticity = 0.0
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")

        # shape.friction = 0.8         # pusher–block grip
        # shape.elasticity = 0.0

        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale), 
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        # [(-60.0, 30), (60.0, 30), (60.0, 0), (-60.0, 0)]   width: 120, height 30

        inertia1 = pymunk.moment_for_poly(mass*1, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        # [(-15.0, 30), (-15.0, 120), (15.0, 120), (15.0, 30)] weight 30, height 90
        inertia2 = pymunk.moment_for_poly(mass*1, vertices=vertices1)
        # inertia2 = pymunk.moment_for_poly(mass*1, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        # body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1.0
        # for s in (shape1, shape2):
        #     s.friction = 0.9
        #     s.elasticity = 0.0
        self.space.add(body, shape1, shape2)
        return body

    # ------------------------------------------------------------------
    # Floor-less planar friction model (Coulomb-like + rotational drag)
    # ------------------------------------------------------------------
    def _apply_planar_friction(self, body: pymunk.Body, dt: float):
        """
        Apply a translational friction force (≈ μ m g, opposite velocity)
        and rotational friction (viscous + Coulomb-like) to a single body.
        Works without a floor shape. Stable for PushT at 100 Hz.
        """
        # --- Translational Coulomb-like friction
        v = body.velocity
        speed = v.length
        if speed > 1e-3:
            F_mag = self.mu_trans * body.mass * self.g_eff
            F = -F_mag * (v / speed)
            com_world = body.local_to_world(body.center_of_gravity)
            print(F)
            body.apply_force_at_world_point(F, com_world)

        # --- Rotational friction: viscous (∝ ω) + Coulomb-like (sign(ω))
        w = body.angular_velocity
        tau = 0.0
        # viscous: scale by moment so tuning is geometry agnostic
        tau += -self.mu_rot_visc * body.moment * w
        if abs(w) > 1e-3:
            tau += -self.tau_rot_coulomb * np.sign(w)
        body.torque += tau


class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self, legacy: bool = False, block_cog=None, damping=None, render_size: int = 96
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
        )
        ws = self.window_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32
                ),
                "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            }
        )
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode="rgb_array")
        agent_pos = np.array(self.agent.position)
        obs = {
            "image": np.moveaxis(img.astype(np.float32) / 255, -1, 0),
            "agent_pos": agent_pos,
            "obs_all": np.array(
                tuple(self.agent.position)
                + tuple(self.block.position)
                + (self.block.angle % (2 * np.pi),)
            ),
        }
        if self.latest_action is not None:
            coord = (np.array(self.latest_action) / 512 * 96).astype(np.int32)
            marker_size = int(8 / 96 * self.render_size)
            thickness = int(1 / 96 * self.render_size)
            cv2.drawMarker(
                img,
                coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness,
            )
        self.render_cache = img
        return obs

    def render(self, mode: str):
        if mode != "rgb_array":
            raise ValueError("PushTImageEnv only supports rgb_array rendering")
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache


__all__ = ["PushTEnv", "PushTImageEnv"]
