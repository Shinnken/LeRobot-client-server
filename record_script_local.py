#!/usr/bin/env python

import base64
import os
import time
from collections import deque
from typing import Any

import numpy as np
import requests

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts, serialize_dict
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ImportError as e:
        raise ImportError(
            "Could not import SO101Follower/SO101FollowerConfig from lerobot. "
            "Check your lerobot version / module path."
        ) from e


FPS = 30
TASK_DESCRIPTION = "Grab the sock and put it in the box"
HF_DATASET_ID = ""
HF_POLICY_ID = "Grigorij/act_right-arm-grab-notebook-2"
# HF_POLICY_ID = "SoSolaris/act_so100_socks"
POLICY_SERVER_URL = "http://100.85.166.124:9000"
RENAME_MAP = {"observation.images.front": "observation.images.image", "observation.images.up": "observation.images.image2"}
# RENAME_MAP = {}
# Action chunking configuration
ACTION_CHUNK_SIZE = 10  # Number of actions to request at once (reduces network latency)
CHUNK_REFILL_THRESHOLD = 0.3  # Refill when queue is below 30% of chunk size


ROBOT_PORT = os.getenv("ROBOT_PORT", "/dev/arm_right")
MAIN_INDEX = int(os.getenv("CAM_MAIN_INDEX", "2"))
RIGHT_INDEX = int(os.getenv("CAM_RIGHT_INDEX", "0"))
SKIP_DATASET_STATS = "1"


def numpy_to_dict(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": base64.b64encode(obj.tobytes()).decode("utf-8"),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    if isinstance(obj, dict):
        return {k: numpy_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_dict(item) for item in obj]
    return obj


def dict_to_numpy(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.frombuffer(base64.b64decode(obj["data"]), dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    if isinstance(obj, dict):
        return {k: dict_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dict_to_numpy(item) for item in obj]
    return obj


def predict_action_remote(
    observation: dict[str, np.ndarray],
    dataset_features: dict,
    dataset_stats: dict | None,
    policy_id: str,
    task: str,
    robot_type: str,
    rename_map: dict[str, str] | None = None,
    chunk_size: int = 1,
) -> dict[str, float] | list[dict[str, float]]:
    dataset_stats_serialized = serialize_dict(dataset_stats) if dataset_stats is not None else {}
    response = requests.post(
        f"{POLICY_SERVER_URL}/predict",
        json={
            "observation": numpy_to_dict(observation),
            "policy_id": policy_id,
            "dataset_features": dataset_features,
            "dataset_stats": dataset_stats_serialized,
            "task": task,
            "robot_type": robot_type,
            "rename_map": rename_map or {},
            "chunk_size": chunk_size,
        },
        timeout=10.0,  # Increased timeout for chunk requests
    )
    response.raise_for_status()
    result = response.json()
    if "action_chunk" in result:
        return [
            {k: float(v.item()) if isinstance(v, np.ndarray) and v.size == 1 else v for k, v in dict_to_numpy(encoded).items()}
            for encoded in result["action_chunk"]
        ]
    return {k: float(v.item()) if isinstance(v, np.ndarray) and v.size == 1 else v for k, v in dict_to_numpy(result["action"]).items()}


def main():
    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id="my_awesome_follower_arm",
            cameras={
                "main": OpenCVCameraConfig(index_or_path=MAIN_INDEX, width=640, height=480, fps=30),
                "right_arm": OpenCVCameraConfig(index_or_path=RIGHT_INDEX, width=640, height=480, fps=30),
            },
        )
    )
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_action_processor, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=robot_observation_processor, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True),
    )

    dataset_stats = None
    if not SKIP_DATASET_STATS:
        dataset = LeRobotDataset.create(
            repo_id=HF_DATASET_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=1,
        )
        dataset_stats = dataset.meta.stats
    listener, events = init_keyboard_listener()
    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    if requests.get(f"{POLICY_SERVER_URL}/health", timeout=2.0).json().get("status") != "ok":
        raise ValueError("Policy server not ready")

    print("Starting inference loop (no recording)...")
    # Action buffer for chunking
    action_buffer = deque()
    
    try:
        while not events["stop_recording"]:
            start_loop_t = time.perf_counter()

            if events.get("exit_early"):
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            # Refill action buffer if needed
            refill_threshold = int(ACTION_CHUNK_SIZE * CHUNK_REFILL_THRESHOLD)
            if len(action_buffer) <= refill_threshold:
                try:
                    chunk_result = predict_action_remote(
                        obs_processed,
                        dataset_features,
                        dataset_stats,
                        HF_POLICY_ID,
                        TASK_DESCRIPTION,
                        robot.name,
                        RENAME_MAP,
                        chunk_size=ACTION_CHUNK_SIZE,
                    )
                    action_buffer.extend(chunk_result[:ACTION_CHUNK_SIZE] if isinstance(chunk_result, list) else [chunk_result])
                except Exception as e:
                    print(f"Error getting action chunk from server: {e}")
                    try:
                        single_action = predict_action_remote(
                            obs_processed,
                            dataset_features,
                            dataset_stats,
                            HF_POLICY_ID,
                            TASK_DESCRIPTION,
                            robot.name,
                            RENAME_MAP,
                            chunk_size=1,
                        )
                        if isinstance(single_action, dict):
                            action_buffer.append(single_action)
                    except Exception as e2:
                        print(f"Error getting single action: {e2}")
                        precise_sleep(max(1 / FPS - (time.perf_counter() - start_loop_t), 0.0))
                        continue

            if len(action_buffer) == 0:
                print("Warning: Action buffer empty, skipping step")
                precise_sleep(max(1 / FPS - (time.perf_counter() - start_loop_t), 0.0))
                continue

            action_values = action_buffer.popleft()
            robot_action_to_send = robot_action_processor((action_values, obs))
            robot.send_action(robot_action_to_send)

            precise_sleep(max(1 / FPS - (time.perf_counter() - start_loop_t), 0.0))
    finally:
        log_say("Stopping inference", blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()
        log_say("Exiting")


if __name__ == "__main__":
    main()
