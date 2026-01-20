#!/usr/bin/env python

import base64
import traceback
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import tempfile
import shutil

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Cache policies by policy_id
policy_cache = {}


class ObservationRequest(BaseModel):
    observation: dict[str, Any]
    policy_id: str
    dataset_features: dict[str, Any]
    dataset_stats: dict[str, Any]
    task: str
    robot_type: str
    rename_map: dict[str, str] = {}
    chunk_size: int = 1  # Number of actions to predict (1 = single action, >1 = chunk)


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


def dict_to_torch(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        arr = np.frombuffer(base64.b64decode(obj["data"]), dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return torch.from_numpy(arr)
    elif isinstance(obj, dict):
        return {k: dict_to_torch(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dict_to_torch(item) for item in obj]
    return obj


def get_or_load_policy(policy_id: str, dataset_features: dict, dataset_stats: dict, rename_map: dict[str, str] = {}):
    cache_key = f"{policy_id}_{hash(frozenset(rename_map.items()))}"
    if cache_key not in policy_cache:
        print(f"Loading new policy {policy_id} with rename_map {rename_map}...")
        tmpdir = tempfile.mkdtemp()
        try:
            print("Creating dataset metadata...")
            dataset_root = Path(tmpdir) / "dataset"
            ds_meta = LeRobotDatasetMetadata.create(
                repo_id="temp",
                fps=30,
                features=dataset_features,
                robot_type=None,
                root=dataset_root,
                use_videos=True,
            )
            print("Converting stats...")
            ds_meta.stats = dict_to_torch(dataset_stats) if dataset_stats else None
            print("Loading policy config...")
            policy_config = PreTrainedConfig.from_pretrained(policy_id)
            policy_config.pretrained_path = policy_id
            print("Making policy...")
            policy = make_policy(policy_config, ds_meta=ds_meta, rename_map=rename_map)
            print("Making processors...")
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy_config,
                pretrained_path=policy_id,
                dataset_stats=rename_stats(ds_meta.stats, rename_map) if ds_meta.stats else None,
                preprocessor_overrides={"device_processor": {"device": policy_config.device}, "rename_observations_processor": {"rename_map": rename_map}},
            )
            device = get_safe_torch_device(policy_config.device)
            use_amp = policy_config.use_amp
            print(f"Policy loaded on device {device}, use_amp={use_amp}")
            policy_cache[cache_key] = {
                "policy": policy,
                "preprocessor": preprocessor,
                "postprocessor": postprocessor,
                "device": device,
                "use_amp": use_amp,
                "dataset_features": dataset_features,
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"Using cached policy {cache_key}")
    return policy_cache[cache_key]


@app.post("/predict")
async def predict(request: ObservationRequest):
    try:
        print(f"Loading policy {request.policy_id}...")
        cache_entry = get_or_load_policy(request.policy_id, request.dataset_features, request.dataset_stats, request.rename_map)
        policy = cache_entry["policy"]
        preprocessor = cache_entry["preprocessor"]
        postprocessor = cache_entry["postprocessor"]
        device = cache_entry["device"]
        use_amp = cache_entry["use_amp"]
        dataset_features = cache_entry["dataset_features"]
        
        print("Converting observation...")
        obs = dict_to_numpy(request.observation)
        print("Building observation frame...")
        obs_frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)
        print("Preparing observation for inference...")
        obs_prepared = prepare_observation_for_inference(obs_frame, device, task=request.task, robot_type=request.robot_type)
        print("Applying preprocessor...")
        obs_processed = preprocessor(obs_prepared)
        print("Running policy inference...")
        with torch.inference_mode(), (torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else torch.no_grad()):
            if request.chunk_size > 1:
                # Predict action chunk
                requested_chunk_size = max(1, int(request.chunk_size))
                action_chunk = policy.predict_action_chunk(obs_processed)  # (B, native_chunk_size, action_dim)
                # Some policies always return their native horizon; crop to the requested chunk size.
                native_chunk_size = int(action_chunk.shape[1])
                chunk_size = min(requested_chunk_size, native_chunk_size)
                action_chunk = action_chunk[:, :chunk_size, :]
                # Apply postprocessor to each action in the chunk
                processed_actions = []
                for i in range(chunk_size):
                    single_action = action_chunk[:, i, :]  # (B, action_dim)
                    processed_action = postprocessor(single_action)
                    processed_actions.append(processed_action)
                # Stack back to (B, chunk_size, action_dim), then remove batch dim
                action_chunk_processed = torch.stack(processed_actions, dim=1).squeeze(0)  # (chunk_size, action_dim)
                # Convert to list of action dicts
                action_chunk_list = []
                for i in range(chunk_size):
                    action_dict = make_robot_action(action_chunk_processed[i], dataset_features)
                    action_chunk_list.append({k: numpy_to_dict(np.array([v], dtype=np.float32)) for k, v in action_dict.items()})
                print(f"Returning action chunk of size {chunk_size}...")
                return {"action_chunk": action_chunk_list, "chunk_size": chunk_size}
            else:
                # Single action (backward compatibility)
                action = policy.select_action(obs_processed)
                print("Applying postprocessor...")
                action = postprocessor(action)
                print("Making robot action...")
                action_values = make_robot_action(action, dataset_features)
                print("Returning action...")
                return {"action": {k: numpy_to_dict(np.array([v], dtype=np.float32)) for k, v in action_values.items()}}
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
