#!/usr/bin/env python

import base64
import time
from collections import deque
from typing import Any

import numpy as np
import requests
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, serialize_dict
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import os
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 6000
TASK_DESCRIPTION = "Grab the sock and put it in the box"
HF_DATASET_ID = "SoSolaris/dqwdwqdwqqewq"
HF_POLICY_ID = "SoSolaris/xvla-your-robot"
# HF_POLICY_ID = "SoSolaris/act_so100_socks"
POLICY_SERVER_URL = "http://192.165.134.28:10380"
# Optional: rename map for observations
# Example: {"observation.images.front": "observation.images.camera1", "observation.images.up": "observation.images.camera2"}
RENAME_MAP = {"observation.images.front": "observation.images.image", "observation.images.up": "observation.images.image2"}
# RENAME_MAP = {}
# Action chunking configuration
ACTION_CHUNK_SIZE = 10  # Number of actions to request at once (reduces network latency)
CHUNK_REFILL_THRESHOLD = 0.3  # Refill when queue is below 30% of chunk size
os.system("rm -rf /Users/antoinemarcel/.cache/huggingface/lerobot/SoSolaris/dqwdwqdwqqewq")


def numpy_to_dict(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": base64.b64encode(obj.tobytes()).decode("utf-8"), "dtype": str(obj.dtype), "shape": obj.shape}
    elif isinstance(obj, dict):
        return {k: numpy_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_dict(item) for item in obj]
    return obj


def dict_to_numpy(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.frombuffer(base64.b64decode(obj["data"]), dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    elif isinstance(obj, dict):
        return {k: dict_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dict_to_numpy(item) for item in obj]
    return obj


def predict_action_remote(observation: dict[str, np.ndarray], dataset_features: dict, dataset_stats: dict | None, policy_id: str, task: str, robot_type: str, rename_map: dict[str, str] | None = None, chunk_size: int = 1) -> dict[str, float] | list[dict[str, float]]:
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
        # Return list of action dicts
        chunk_list = []
        for action_dict_encoded in result["action_chunk"]:
            action_dict = {k: float(v.item()) if isinstance(v, np.ndarray) and v.size == 1 else v for k, v in dict_to_numpy(action_dict_encoded).items()}
            chunk_list.append(action_dict)
        return chunk_list
    else:
        # Single action (backward compatibility)
        return {k: float(v.item()) if isinstance(v, np.ndarray) and v.size == 1 else v for k, v in dict_to_numpy(result["action"]).items()}


def main():
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/cu.usbmodem58760433451",
        id="my_awesome_follower_arm",
        cameras={"front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30), "up": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=60)},
    ))
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_action_processor, initial_features=create_initial_features(action=robot.action_features), use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=robot_observation_processor, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True),
    )
    dataset = LeRobotDataset.create(repo_id=HF_DATASET_ID, fps=FPS, features=dataset_features, robot_type=robot.name, use_videos=True, image_writer_threads=4)
    listener, events = init_keyboard_listener()
    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    if not requests.get(f"{POLICY_SERVER_URL}/health", timeout=2.0).json().get("status") == "ok":
        raise ValueError("Policy server not ready")
    print("Starting record loop...")
    # Action buffer for chunking
    action_buffer = deque()
    
    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
                log_say(f"Recording episode {recorded_episodes + 1} of {NUM_EPISODES}")
                timestamp = 0
                start_episode_t = time.perf_counter()
                while timestamp < EPISODE_TIME_SEC:
                    start_loop_t = time.perf_counter()
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
                    obs = robot.get_observation()
                    obs_processed = robot_observation_processor(obs)
                    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    
                    # Refill action buffer if needed
                    refill_threshold = int(ACTION_CHUNK_SIZE * CHUNK_REFILL_THRESHOLD)
                    if len(action_buffer) <= refill_threshold:
                        try:
                            print(f"Refilling action buffer (current size: {len(action_buffer)})...")
                            chunk_result = predict_action_remote(
                                obs_processed, 
                                dataset.features, 
                                dataset.meta.stats, 
                                HF_POLICY_ID, 
                                TASK_DESCRIPTION, 
                                robot.name, 
                                RENAME_MAP,
                                chunk_size=ACTION_CHUNK_SIZE
                            )
                            if isinstance(chunk_result, list):
                                # Add at most ACTION_CHUNK_SIZE actions from chunk to buffer
                                action_buffer.extend(chunk_result[:ACTION_CHUNK_SIZE])
                                print(f"Action buffer refilled to {len(action_buffer)} actions")
                            else:
                                # Single action (fallback)
                                action_buffer.append(chunk_result)
                        except Exception as e:
                            print(f"Error getting action chunk from server: {e}")
                            # Fallback to single action
                            try:
                                single_action = predict_action_remote(
                                    obs_processed, 
                                    dataset.features, 
                                    dataset.meta.stats, 
                                    HF_POLICY_ID, 
                                    TASK_DESCRIPTION, 
                                    robot.name, 
                                    RENAME_MAP,
                                    chunk_size=1
                                )
                                if isinstance(single_action, dict):
                                    action_buffer.append(single_action)
                            except Exception as e2:
                                print(f"Error getting single action: {e2}")
                                continue
                    
                    # Get next action from buffer
                    if len(action_buffer) > 0:
                        action_values = action_buffer.popleft()
                    else:
                        print("Warning: Action buffer empty, skipping frame")
                        continue
                    
                    robot_action_to_send = robot_action_processor((action_values, obs))
                    robot.send_action(robot_action_to_send)
                    action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                    dataset.add_frame({**observation_frame, **action_frame, "task": TASK_DESCRIPTION})
                    precise_sleep(max(1 / FPS - (time.perf_counter() - start_loop_t), 0.0))
                    timestamp = time.perf_counter() - start_episode_t
                if events["rerecord_episode"]:
                    log_say("Re-recording episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue
                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", blocking=True)
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()
        # dataset.push_to_hub()
        log_say("Exiting")


if __name__ == "__main__":
    main()
