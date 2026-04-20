#!/usr/bin/env python3
"""
run_ur10_policy.py — OpenPi π0 inference on UR10 CB3 with PincOpen gripper.

Usage:
    python run_ur10_policy.py \
        --checkpoint checkpoints/pi0_ur10_lora_corrected/pi0_lora_ur10_corrected/4999 \
        --prompt "grasp the object and place it in the box" \
        [--num-episodes 1] [--max-steps 300] [--dry-run]
"""

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np

# ── OpenPi imports ────────────────────────────────────────────────────────────
import openpi.models.pi0_config as pi0_config
import openpi.serving.websocket_policy_server as _server   # noqa: needed for policy runner
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.inference as _inference

# ── Robot imports ─────────────────────────────────────────────────────────────
import rtde_control
import rtde_receive

# GripperController from your existing robot package
import sys
sys.path.insert(0, "/home_local/rudra_1/rudra")          # adjust if needed
from lerobot_ur10.robots.ur10.pincopen_gripper import GripperController
from lerobot_ur10.robots.ur10.config_ur10 import UR10Config

# RealSense
import pyrealsense2 as rs
import cv2


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────── Constants ───────────────────────────────────

ROBOT_IP            = "192.168.100.3"
HOME_JOINTS_DEG     = [0, -90, 90, -90, -90, 90]
HOME_JOINTS_RAD     = [math.radians(d) for d in HOME_JOINTS_DEG]

GRIPPER_PORT        = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4W75-if00-port0"
GRIPPER_BAUD        = 1_000_000
GRIPPER_DXL_ID      = 0
GRIPPER_OPEN_ANGLE  = 285.0
GRIPPER_CLOSE_ANGLE = 166.0

CAM_WRIST_SERIAL    = "923322071837"   # D435i
CAM_TOP_SERIAL      = "204322061013"   # D415
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30

# servoJ params (CB3 @ 125 Hz)
SERVOJ_T          = 1.0 / 125
SERVOJ_LOOKAHEAD  = 0.1
SERVOJ_GAIN       = 200
SERVOJ_SPEED      = 0.1
SERVOJ_ACC        = 0.1

# Policy
ACTION_HORIZON    = 50   # pi0 default; only first EXEC_HORIZON actions executed
EXEC_HORIZON      = 8    # receding-horizon execution (action chunking)
PROMPT            = "grasp the object and place it in the box"


# ──────────────────────────── Camera helpers ─────────────────────────────────

class RealSenseCamera:
    def __init__(self, serial: str, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()

    def start(self):
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(self.config)
        # warmup
        for _ in range(10):
            self.pipeline.wait_for_frames()
        logger.info(f"Camera {self.serial} started")

    def read(self) -> np.ndarray:
        """Returns uint8 RGB (H, W, 3)."""
        frames = self.pipeline.wait_for_frames()
        color  = frames.get_color_frame()
        bgr    = np.asanyarray(color.get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.pipeline.stop()


# ──────────────────────────── Robot wrapper ──────────────────────────────────

class UR10PolicyRunner:
    def __init__(self, args):
        self.args = args

        # RTDE
        logger.info(f"Connecting to UR10 at {ROBOT_IP} …")
        self.ctrl = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        logger.info("RTDE connected")

        # Gripper
        self.gripper = GripperController(
            port=GRIPPER_PORT,
            baud=GRIPPER_BAUD,
            dxl_id=GRIPPER_DXL_ID,
            open_angle=GRIPPER_OPEN_ANGLE,
            close_angle=GRIPPER_CLOSE_ANGLE,
            default_speed=1.0,
            default_torque=0.2,
        )
        self._last_gripper_cmd      = -1.0
        self._last_gripper_cmd_time = 0.0

        # Cameras
        self.cam_wrist = RealSenseCamera(CAM_WRIST_SERIAL)
        self.cam_top   = RealSenseCamera(CAM_TOP_SERIAL)
        self.cam_wrist.start()
        self.cam_top.start()

        # Policy
        self.policy = self._load_policy(args.checkpoint)

    # ── Policy loading ────────────────────────────────────────────────────

    def _load_policy(self, checkpoint_dir: str):
        """Loads the trained OpenPi policy from a checkpoint directory."""
        checkpoint_path = Path(checkpoint_dir)
        # Find the config name from the checkpoint path structure:
        # checkpoints/<config_name>/<exp_name>/<step>/
        config_name = checkpoint_path.parts[-3]   # e.g. "pi0_ur10_lora_corrected"

        logger.info(f"Loading config: {config_name}")
        train_config = _config.get_config(config_name)
        model_config = train_config.model

        policy = _inference.create_policy(
            train_config,
            checkpoint_path=checkpoint_path,
        )
        logger.info(f"Policy loaded from {checkpoint_path}")
        return policy

    # ── Observation builder ───────────────────────────────────────────────

    def _get_obs(self) -> dict:
        """Builds the observation dict in the format UR10Inputs expects."""
        joints       = self.recv.getActualQ()          # [j0..j5] radians
        gripper_norm = self.gripper.get_pos_normalized()  # [0=open, 1=closed]

        state = np.array(joints + [gripper_norm], dtype=np.float32)  # (7,)

        wrist_img = self.cam_wrist.read()   # uint8 (H,W,3) RGB
        top_img   = self.cam_top.read()     # uint8 (H,W,3) RGB

        return {
            # Keys must match your RepackTransform in LeRobotUR10DataConfig
            "observation.state":                  state,
            "observation.images.cam_high":         top_img,
            "observation.images.cam_right_wrist":  wrist_img,
            "prompt":                              self.args.prompt,
        }

    # ── Action execution ──────────────────────────────────────────────────

    def _send_action(self, action: np.ndarray):
        """
        action: (7,) float32 — [j0..j5 radians, gripper_normalized]
        Mirrors ur10.py send_action() exactly.
        """
        assert len(action) == 7, f"Expected 7-dim action, got {len(action)}"

        t_start = self.ctrl.initPeriod()

        goal_joints = action[:6].tolist()
        self.ctrl.servoJ(
            goal_joints,
            SERVOJ_SPEED,
            SERVOJ_ACC,
            SERVOJ_T,
            SERVOJ_LOOKAHEAD,
            SERVOJ_GAIN,
        )

        # Gripper — throttled, matches ur10.py logic
        gripper_cmd = float(np.clip(action[6], 0.0, 1.0))
        now = time.monotonic()
        delta = abs(gripper_cmd - self._last_gripper_cmd)
        if delta >= 0.002 and (now - self._last_gripper_cmd_time) >= 0.1:
            self.gripper.set_pos_normalized_async(gripper_cmd)
            self._last_gripper_cmd      = gripper_cmd
            self._last_gripper_cmd_time = now

        self.ctrl.waitPeriod(t_start)

    # ── Home / safety ─────────────────────────────────────────────────────

    def go_home(self, speed=0.3, acc=0.3):
        logger.info(f"Moving to home: {HOME_JOINTS_DEG} deg")
        self.ctrl.servoStop()        # exit servoJ mode before moveJ
        time.sleep(0.1)
        self.ctrl.moveJ(HOME_JOINTS_RAD, speed, acc)
        logger.info("At home position")

    # ── Episode loop ──────────────────────────────────────────────────────

    def run_episode(self, episode_idx: int):
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode_idx + 1} — prompt: '{self.args.prompt}'")
        logger.info(f"{'='*50}")

        self.go_home()
        input("Press ENTER when scene is ready …")

        step = 0
        action_chunk: np.ndarray | None = None
        chunk_step  = EXEC_HORIZON   # force inference on first step

        try:
            while step < self.args.max_steps:
                loop_start = time.perf_counter()

                # ── Re-infer every EXEC_HORIZON steps (action chunking)
                if chunk_step >= EXEC_HORIZON:
                    obs = self._get_obs()
                    if self.args.dry_run:
                        action_chunk = np.zeros((ACTION_HORIZON, 7), dtype=np.float32)
                        logger.info("[DRY RUN] skipping policy inference")
                    else:
                        result       = self.policy.infer(obs)
                        # result["actions"] shape: (ACTION_HORIZON, 7) after UR10Outputs trim
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        logger.info(
                            f"Step {step:4d} | re-inferred chunk "
                            f"| action[0] = {action_chunk[0].round(3)}"
                        )
                    chunk_step = 0

                # ── Execute current chunk action
                action = action_chunk[chunk_step]   # (7,)
                self._send_action(action)
                chunk_step += 1
                step       += 1

                # ── Timing info
                elapsed = (time.perf_counter() - loop_start) * 1e3
                logger.debug(f"Step {step} loop: {elapsed:.1f} ms")

        except KeyboardInterrupt:
            logger.info("Episode interrupted by user")
        finally:
            self.ctrl.servoStop()
            logger.info(f"Episode {episode_idx + 1} done ({step} steps)")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        logger.info("Shutting down …")
        try:
            self.ctrl.servoStop()
            self.go_home()
        except Exception as e:
            logger.warning(f"Error during shutdown home: {e}")
        self.ctrl.disconnect()
        self.recv.disconnect()
        self.gripper.disconnect()
        self.cam_wrist.stop()
        self.cam_top.stop()
        logger.info("All devices disconnected")


# ────────────────────────────────── Main ─────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint step dir, e.g. "
             "checkpoints/pi0_ur10_lora_corrected/pi0_lora_ur10_corrected/4999",
    )
    p.add_argument("--prompt",       default=PROMPT)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-steps",    type=int, default=300,
                   help="Hard cutoff per episode (125 Hz → 300 steps ≈ 2.4 s/chunk × receding)")
    p.add_argument("--exec-horizon", type=int, default=EXEC_HORIZON,
                   help="How many actions to execute before re-inferring (action chunking window)")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect hardware but skip policy inference (zeros actions)")
    return p.parse_args()


def main():
    args   = parse_args()
    runner = UR10PolicyRunner(args)

    try:
        for ep in range(args.num_episodes):
            runner.run_episode(ep)
            if ep < args.num_episodes - 1:
                ans = input("\nRun another episode? [Y/n]: ")
                if ans.strip().lower() == "n":
                    break
    finally:
        runner.close()


if __name__ == "__main__":
    main()