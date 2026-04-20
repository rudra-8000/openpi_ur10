#!/usr/bin/env python3
"""
UR10 real-robot client for OpenPi policy inference.

Connects to a remote OpenPi WebSocket server, streams observations, and
executes returned actions on the UR10 CB3 via RTDE + PincOpen gripper.

Usage (robot workstation):

  # 1 rollout, 200 steps, record video
  python examples/ur10/client_ur10.py \
      --server-host 10.245.91.19 --server-port 8765 \
      --max-steps 200 \
      --video-dir ./videos

  # 3 rollouts, auto-reset between each
  python examples/ur10/client_ur10.py \
      --server-host 10.245.91.19 --server-port 8765 \
      --max-steps 200 --num-rollouts 3 \
      --reset-between-rollouts \
      --video-dir ./videos

  # Dry-run: print actions without moving the robot
  python examples/ur10/client_ur10.py \
      --server-host 10.245.91.19 --server-port 8765 \
      --max-steps 10 --dry-run

Dependencies (robot workstation only):
  pip install ur-rtde websockets msgpack-numpy pyrealsense2 dynamixel-sdk
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Home position  (degrees → radians)
# ──────────────────────────────────────────────────────────────────────────────
HOME_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 90.0)
HOME_RAD = tuple(np.deg2rad(d) for d in HOME_DEG)

# Camera serials — match your hardware
CAM_SERIALS = {
    "cam_high":        "204322061013",   # D415 top
    "cam_right_wrist": "923322071837",   # D435i wrist
}

# ──────────────────────────────────────────────────────────────────────────────
# Robot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_robot(args: argparse.Namespace):
    """Construct a UR10 robot with cameras attached."""
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.cameras import make_cameras_from_configs
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.lerobot_robot_ur10 import UR10Config  # adjust import to your package layout

    cam_cfgs = {
        name: RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=args.fps,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        )
        for name, serial in CAM_SERIALS.items()
    }

    robot_cfg = UR10Config(ip=args.ur_ip)
    robot = make_robot_from_config(robot_cfg)
    robot.cameras = make_cameras_from_configs(cam_cfgs)
    return robot


def smooth_move_home(robot, home_rad=HOME_RAD, steps: int = 100, duration: float = 5.0) -> None:
    """
    Linearly interpolate from current joint positions to home over `duration` seconds.
    Uses servoJ so the motion stays within the RTDE control loop.
    """
    import rtde_receive

    current = list(robot.rtde_rec.getActualQ())
    target  = list(home_rad)
    dt      = duration / steps
    logger.info("Moving to home: %s (%.1f s)", [f"{np.rad2deg(v):.1f}" for v in target], duration)
    for i in range(1, steps + 1):
        alpha = i / steps
        interp = [current[j] + alpha * (target[j] - current[j]) for j in range(6)]
        robot.rtde_ctrl.servoJ(
            interp,
            robot.speed,
            robot.acc,
            robot.servoj_t,
            robot.servoj_lookahead,
            robot.servoj_gain,
        )
        time.sleep(dt)
    # Final blocking moveJ to guarantee convergence
    robot.rtde_ctrl.moveJ(list(home_rad), speed=0.5, acceleration=0.3)
    logger.info("Home reached.")


# ──────────────────────────────────────────────────────────────────────────────
# Observation / action serialisation
# ──────────────────────────────────────────────────────────────────────────────

def _obs_to_payload(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert robot.get_observation() dict into the msgpack payload the
    OpenPi server expects.

    OpenPi server (websocket_policy_server.py) receives a flat dict where:
      - scalar joint values:  "joint_0" .. "joint_5", "gripper"
      - camera images:        camera names as keys  (H, W, 3) uint8

    The server's UR10Policy.infer() then builds:
      observation.state  = [joint_0..5, gripper]  (shape 7, padded to 32)
      observation.images = {"cam_high": ..., "cam_right_wrist": ...}
    """
    out: dict[str, Any] = {}
    for k, v in raw_obs.items():
        if isinstance(v, np.ndarray):
            out[k] = np.ascontiguousarray(v, dtype=np.uint8 if v.dtype == np.uint8 else np.float32)
        else:
            out[k] = float(v)
    return out


def _unpack_action(resp: dict) -> dict[str, float]:
    """
    The OpenPi server returns {"action": {"joint_0": ..., ..., "gripper": ...}}.
    Validate and return the flat action dict.
    """
    action = resp.get("action")
    if action is None:
        raise RuntimeError(f"No 'action' in server response. Keys: {list(resp.keys())}")
    # action may come back as a dict of numpy scalars — normalise to float
    return {k: float(v) for k, v in action.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Video recording
# ──────────────────────────────────────────────────────────────────────────────

def _side_by_side(raw_obs: dict, keys: list[str]) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    frames = [raw_obs[k] for k in keys if k in raw_obs and isinstance(raw_obs[k], np.ndarray)]
    if not frames:
        return None
    h = max(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        if f.shape[0] != h:
            w = int(round(f.shape[1] * h / f.shape[0]))
            f = cv2.resize(f, (w, h))
        resized.append(f)
    return np.concatenate(resized, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    try:
        import msgpack_numpy
        from websockets.asyncio.client import connect
    except ImportError as e:
        raise ImportError("pip install websockets msgpack-numpy") from e

    packer = msgpack_numpy.Packer()
    uri = f"ws://{args.server_host}:{args.server_port}/"

    if args.dry_run:
        logger.warning("DRY-RUN mode: robot will NOT be connected or moved.")
        robot = None
    else:
        robot = _build_robot(args)
        robot.connect()
        logger.info("Robot connected.")

    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    num_rollouts = max(1, args.num_rollouts)
    if num_rollouts > 1 and args.max_steps <= 0:
        raise ValueError("--num-rollouts > 1 requires --max-steps > 0")

    control_dt = args.control_dt if args.control_dt else 1.0 / args.fps
    logger.info("Control dt: %.4f s (%.1f Hz)", control_dt, 1.0 / control_dt)

    try:
        async with connect(uri, max_size=None, compression=None) as ws:
            # Handshake
            meta_raw = await ws.recv()
            meta = msgpack_numpy.unpackb(meta_raw)
            logger.info("Server handshake: %s", meta)
            if meta.get("protocol") not in ("openpi_v1", "lerobot_policy_v1"):
                logger.warning("Unexpected protocol %r — continuing.", meta.get("protocol"))

            loop = asyncio.get_running_loop()

            for rollout in range(num_rollouts):
                logger.info("━━━ Rollout %d / %d ━━━", rollout + 1, num_rollouts)

                # Reset policy hidden state between rollouts
                if rollout > 0 and args.reset_between_rollouts:
                    await ws.send(packer.pack({"__ctrl__": "reset"}))
                    ack = msgpack_numpy.unpackb(await ws.recv())
                    logger.info("Policy reset ack: %s", ack)

                # Move to home
                if robot is not None:
                    smooth_move_home(robot)
                    time.sleep(2.0)   # settle cameras + gripper

                # Video writer
                writer = None
                vid_path = video_dir / f"rollout_{rollout:04d}.mp4" if video_dir else None

                step = 0
                while True:
                    if args.max_steps > 0 and step >= args.max_steps:
                        logger.info("Reached max_steps=%d, ending rollout.", args.max_steps)
                        break

                    t0 = loop.time()

                    # ── Observation ───────────────────────────────────────
                    if robot is not None:
                        raw_obs = await asyncio.to_thread(robot.get_observation)
                    else:
                        # Fake obs for dry-run
                        raw_obs = {
                            **{f"joint_{i}": float(HOME_RAD[i]) for i in range(6)},
                            "gripper": 0.0,
                            "cam_high":        np.zeros((480, 640, 3), dtype=np.uint8),
                            "cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                        }

                    # ── Send to server ────────────────────────────────────
                    payload = _obs_to_payload(raw_obs)
                    msg: dict[str, Any] = {"observation": payload}
                    if args.task:
                        msg["task"] = args.task
                    await ws.send(packer.pack(msg))

                    # ── Receive action ────────────────────────────────────
                    resp = msgpack_numpy.unpackb(await ws.recv())
                    if not isinstance(resp, dict):
                        raise RuntimeError(f"Bad response: {type(resp)}")
                    if "error" in resp:
                        raise RuntimeError(f"Server error: {resp['error']}")

                    action = _unpack_action(resp)

                    if args.log_every > 0 and step % args.log_every == 0:
                        st = resp.get("server_timing") or {}
                        joints = [f"{np.rad2deg(action.get(f'joint_{i}', 0)):.1f}°" for i in range(6)]
                        logger.info(
                            "[r%d s%d] joints=%s gripper=%.3f infer_ms=%.1f",
                            rollout, step, joints,
                            action.get("gripper", 0),
                            st.get("infer_ms", -1),
                        )

                    # ── Record video ──────────────────────────────────────
                    if vid_path is not None:
                        import cv2
                        composite = _side_by_side(raw_obs, ["cam_high", "cam_right_wrist"])
                        if composite is not None:
                            if writer is None:
                                h, w = composite.shape[:2]
                                fps_out = max(1.0, 1.0 / control_dt)
                                writer = cv2.VideoWriter(
                                    str(vid_path),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps_out,
                                    (w, h),
                                )
                                logger.info("Recording %s (%dx%d @ %.0f fps)", vid_path.name, w, h, fps_out)
                            writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

                    # ── Execute action ────────────────────────────────────
                    if robot is not None:
                        await asyncio.to_thread(robot.send_action, action)

                    step += 1

                    # ── Timing ────────────────────────────────────────────
                    elapsed = loop.time() - t0
                    sleep_t = control_dt - elapsed
                    if sleep_t > 1e-3:
                        await asyncio.sleep(sleep_t)
                    elif sleep_t < -0.005:
                        logger.debug("Step overran by %.1f ms", -sleep_t * 1e3)

                if writer is not None:
                    writer.release()
                    logger.info("Saved %s (%d frames)", vid_path, step)

    finally:
        if robot is not None and robot.is_connected:
            logger.info("Disconnecting robot...")
            robot.disconnect()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(description="OpenPi UR10 inference client")
    p.add_argument("--server-host", default="10.245.91.19", help="GPU server hostname / IP")
    p.add_argument("--server-port", type=int, default=8765, help="WebSocket port")
    p.add_argument("--ur-ip", default="192.168.100.3", help="UR10 controller IP")
    p.add_argument("--fps", type=int, default=30, help="Camera / control rate Hz")
    p.add_argument("--control-dt", type=float, default=None,
                   help="Override control timestep (seconds). Default: 1/fps")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Steps per rollout (0 = run until Ctrl-C)")
    p.add_argument("--num-rollouts", type=int, default=1, help="Number of rollouts")
    p.add_argument("--reset-between-rollouts", action="store_true",
                   help="Send policy reset message between rollouts")
    p.add_argument("--task", default="grasp the object and place it in the box",
                   help="Language prompt sent to the policy")
    p.add_argument("--video-dir", default="", help="Save per-rollout mp4 videos here")
    p.add_argument("--log-every", type=int, default=30,
                   help="Log joint angles / timing every N steps (0 to disable)")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect to server but do not move the robot (use fake observations)")
    args = p.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


if __name__ == "__main__":
    main()