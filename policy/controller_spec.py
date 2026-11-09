"""Plain-math statement of what the authored weights are meant to compute.

This is the specification, not the policy: nothing here runs at evaluation
time. It is used by policy/test_weights.py to check that the network's forward
pass reproduces these formulas.
"""

from __future__ import annotations

from gen_weights import (BALL_SIZE, BRICK_BOTTOM, DEADBAND, LAND_Y,
    LEAD_FRAMES, X_MAX)

NOOP, LEFT, RIGHT = 0, 1, 2


def fold(u: float) -> float:
    """Mirror an unbounded x coordinate back into [0, X_MAX] (wall bounces)."""
    period = 2.0 * X_MAX
    u = u % period
    return period - u if u > X_MAX else u


def landing_x(x: float, y: float, vx: float, vy: float) -> float:
    """Ball left-edge x when its top reaches LAND_Y, for a falling ball."""
    return fold(x + vx * (LAND_Y - y) / vy)


def decide(p: float, pw: float, x: float, y: float, vx: float, vy: float):
    """Returns (action, mode, signed error in px that drives the action)."""
    q = p + pw / 2.0 - BALL_SIZE / 2.0   # ball x that is centred on the paddle
    if vy > 0 and y >= BRICK_BOTTOM:
        mode = "intercept"
        error = landing_x(x, y, vx, vy) - q
    else:
        mode = "track"
        error = (x - q) + LEAD_FRAMES * vx
    if error > DEADBAND:
        return RIGHT, mode, error
    if error < -DEADBAND:
        return LEFT, mode, error
    return NOOP, mode, error
