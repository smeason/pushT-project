"""PushT environment variant for rectangular and other convex polygon blocks."""

import numpy as np
import pygame
import pymunk
from shapely.geometry import Polygon

from gym_pusht.envs.pusht import PushTEnv


class ArbitraryShapePushEnv(PushTEnv):
    """Push a square, rectangle, or user-defined convex polygon.

    ``vertices`` are local polygon coordinates. They are centered around their
    area centroid so the block pose and goal pose refer to the shape center.
    Concave polygons are rejected; use a convex outline or decompose a concave
    shape into multiple convex shapes before adding that support.
    """

    def __init__(
        self,
        *,
        shape="rectangle",
        width=50.0,
        height=100.0,
        vertices=None,
        rotation_symmetry_order=None,
        **kwargs,
    ):
        self.shape_kind = str(shape).lower()
        self.block_width = float(width)
        self.block_height = float(height)
        if not np.isfinite(self.block_width) or self.block_width <= 0:
            raise ValueError("width must be a positive finite number")
        if not np.isfinite(self.block_height) or self.block_height <= 0:
            raise ValueError("height must be a positive finite number")

        if self.shape_kind == "square":
            side = self.block_width
            vertices = [
                (-side / 2, -side / 2),
                (side / 2, -side / 2),
                (side / 2, side / 2),
                (-side / 2, side / 2),
            ]
            default_symmetry = 4
        elif self.shape_kind == "rectangle":
            half_width, half_height = self.block_width / 2, self.block_height / 2
            vertices = [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            ]
            default_symmetry = 2
        elif self.shape_kind == "polygon":
            if vertices is None:
                raise ValueError("vertices are required when shape='polygon'")
            default_symmetry = 1
        else:
            raise ValueError("shape must be 'square', 'rectangle', or 'polygon'")

        polygon_vertices = np.asarray(vertices, dtype=np.float64)
        if polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2 or len(polygon_vertices) < 3:
            raise ValueError("vertices must contain at least three (x, y) points")
        if not np.isfinite(polygon_vertices).all():
            raise ValueError("vertices must contain only finite coordinates")
        polygon = Polygon(polygon_vertices)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("vertices must describe a valid polygon with positive area")
        if not np.isclose(polygon.area, polygon.convex_hull.area, rtol=1e-6, atol=1e-8):
            raise ValueError("Pymunk polygon blocks must be convex")

        centroid = np.asarray(polygon.centroid.coords[0], dtype=np.float64)
        polygon_vertices -= centroid
        self.block_vertices = [tuple(point) for point in polygon_vertices]

        symmetry = default_symmetry if rotation_symmetry_order is None else rotation_symmetry_order
        if not isinstance(symmetry, int) or symmetry < 1:
            raise ValueError("rotation_symmetry_order must be a positive integer")
        self.rotation_symmetry_order = symmetry

        super().__init__(**kwargs)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False

        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        self.agent = self.add_circle(self.space, (256, 400), 15)

        mass = 1.0
        moment = pymunk.moment_for_poly(mass, vertices=self.block_vertices)
        self.block = pymunk.Body(mass, moment)
        block_shape = pymunk.Poly(self.block, self.block_vertices)
        block_shape.color = pygame.Color("LightSlateGray")
        self.block.center_of_gravity = (0, 0)
        self.block.position = (256, 300)
        self.block.angle = 0.0
        self._block_shapes = [block_shape]
        self.space.add(self.block, block_shape)

        self.goal_pose = np.array([256.0, 256.0, 0.0])
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0
