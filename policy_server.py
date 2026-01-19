import logging
import io
import socket
import struct
import time

import torch
try:
    import torch._dynamo as torch_dynamo
except Exception:  # pragma: no cover
    torch_dynamo = None
from PIL import Image
from torchvision import transforms
from lerobot.processor import create_transition, transition_to_batch, TransitionKey
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PolicyServer:
    def __init__(self):
        self.policy_path: str = "Grigorij/pi05_right-arm-grab-notebook"
        self.policy_type: str = "pi05"
        self.host: str = "0.0.0.0"
        self.port: int = 9000
        self.device = torch.device("cuda")
        self.action_chunk_size: int | None = None
        self._max_text_len: int | None = None

        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch_dynamo is not None:
            torch_dynamo.config.suppress_errors = True

        self._ensure_siglip_check()
        from lerobot.policies.factory import get_policy_class

        # Load model
        logging.info(f"Loading policy from: {self.policy_path}")
        policy_config = PreTrainedConfig.from_pretrained(
            self.policy_path,
            cli_overrides=[
                "--dtype=float32",
                "--compile_model=false",
                "--use_amp=false",
                f"--device={self.device.type}",
            ],
        )
        policy_class = get_policy_class(self.policy_type)
        self.policy = policy_class.from_pretrained(self.policy_path, config=policy_config)
        self.policy.to(self.device)
        self.policy = self.policy.float()
        self.policy.eval()
        logging.info(f"Policy loaded successfully on {self.device}")
        self._max_text_len = self._infer_max_text_len(self.policy.config, self.policy)
        if self._max_text_len is not None:
            logging.info("Using max text length: %s", self._max_text_len)

        self.image_transform = transforms.Compose([transforms.ToTensor()])

        self.preprocessor = None
        self.postprocessor = None
        try:
            from lerobot.processor import DataProcessorPipeline

            self.preprocessor = DataProcessorPipeline.from_pretrained(
                self.policy_path, config_filename="policy_preprocessor.json"
            )
            self.postprocessor = DataProcessorPipeline.from_pretrained(
                self.policy_path, config_filename="policy_postprocessor.json"
            )
        except Exception as exc:
            logging.warning(
                "Falling back to default PI05 processors (no pretrained processor configs found): %s",
                exc,
            )
            self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(self.policy.config)

    @staticmethod
    def _ensure_siglip_check():
        """Ensure SigLIP check does not block policy init."""
        try:
            from transformers.models.siglip import check as siglip_check

            if hasattr(siglip_check, "check_whether_transformers_replace_is_installed_correctly"):
                siglip_check.check_whether_transformers_replace_is_installed_correctly = lambda: True
        except Exception:
            import sys
            import types

            mod = types.ModuleType("transformers.models.siglip.check")
            mod.check_whether_transformers_replace_is_installed_correctly = lambda: True
            sys.modules["transformers.models.siglip.check"] = mod

    @staticmethod
    def _infer_max_text_len(config, policy=None) -> int | None:
        """Best-effort extraction of max text sequence length from policy config or model."""
        def _pick_int(obj, names):
            for name in names:
                value = getattr(obj, name, None)
                if isinstance(value, int) and value > 0:
                    return value
            return None

        names = [
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "max_length",
            "seq_length",
        ]
        # Check direct config
        value = _pick_int(config, names)
        if value is not None:
            return value

        # Check nested config objects if present
        nested_attrs = [
            "text_config",
            "language_config",
            "gemma_config",
            "paligemma_config",
            "model_config",
            "gemma_expert_config",
            "expert_config",
        ]
        for attr in nested_attrs:
            nested = getattr(config, attr, None)
            if nested is None:
                continue
            value = _pick_int(nested, names)
            if value is not None:
                return value

        # Check model objects if available
        if policy is not None:
            model_attrs = [
                "model",
                "paligemma_with_expert",
                "gemma_expert",
            ]

            def _walk(obj):
                if obj is None:
                    return None
                value = _pick_int(obj, names)
                if value is not None:
                    return value
                conf = getattr(obj, "config", None)
                value = _pick_int(conf, names)
                if value is not None:
                    return value
                return None

            # Try common nested paths
            value = _walk(policy)
            if value is not None:
                return value

            current = policy
            for attr in model_attrs:
                current = getattr(current, attr, None)
                value = _walk(current)
                if value is not None:
                    return value
                # Attempt subfields for known stacks
                if current is not None:
                    for sub_attr in ("model", "gemma_expert", "text_model", "language_model"):
                        value = _walk(getattr(current, sub_attr, None))
                        if value is not None:
                            return value
        return None

    def _coerce_language_batch(self, batch):
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        if tokens is None or masks is None:
            raise RuntimeError("Missing language tokens/masks in batch")

        if self._max_text_len is None:
            self._max_text_len = self._infer_gemma_max_positions()
            if self._max_text_len is not None:
                logging.info("Inferred max text length from Gemma model: %s", self._max_text_len)

        if self._max_text_len is not None and tokens.shape[1] > self._max_text_len:
            tokens = tokens[:, : self._max_text_len]
            masks = masks[:, : self._max_text_len]
            batch[OBS_LANGUAGE_TOKENS] = tokens
            batch[OBS_LANGUAGE_ATTENTION_MASK] = masks

        if tokens.shape[:2] != masks.shape[:2]:
            target_len = tokens.shape[1]
            current_len = masks.shape[1]
            if current_len < target_len:
                pad = torch.zeros(
                    masks.shape[0],
                    target_len - current_len,
                    dtype=masks.dtype,
                    device=masks.device,
                )
                masks = torch.cat([masks, pad], dim=1)
            else:
                masks = masks[:, :target_len]
            batch[OBS_LANGUAGE_ATTENTION_MASK] = masks

        if masks.dtype != torch.bool:
            batch[OBS_LANGUAGE_ATTENTION_MASK] = masks.to(dtype=torch.bool)

        logging.debug(
            "Language tokens/masks shapes: tokens=%s masks=%s max_text_len=%s",
            tuple(tokens.shape),
            tuple(batch[OBS_LANGUAGE_ATTENTION_MASK].shape),
            self._max_text_len,
        )

    def _infer_gemma_max_positions(self) -> int | None:
        """Fallback: read Gemma max_position_embeddings from the loaded model."""
        candidates = [
            ("model", "paligemma_with_expert", "gemma_expert", "model", "config"),
            ("model", "paligemma_with_expert", "gemma_expert", "config"),
            ("model", "gemma_expert", "model", "config"),
            ("model", "gemma_expert", "config"),
        ]

        for path in candidates:
            obj = self.policy
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
            value = getattr(obj, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def process_observation(self, timestamp, image_data_main, image_data_right, joint_states, task_name):
        # Prepare image
        pil_image = Image.open(io.BytesIO(image_data_main)).convert("RGB")
        pil_image_right = Image.open(io.BytesIO(image_data_right)).convert("RGB")
        image_tensor_main = self.image_transform(pil_image)
        image_tensor_right = self.image_transform(pil_image_right)

        # Prepare state
        state_tensor = torch.tensor(joint_states, dtype=torch.float32)

        # Create observation dict
        observation = {
            "observation.images.main": image_tensor_main,
            "observation.images.right_arm": image_tensor_right,
            "observation.state": state_tensor,
        }

        if self.preprocessor is None:
            raise RuntimeError("Policy preprocessor is not available")

        transition = create_transition(
            observation=observation,
            complementary_data={"task": [task_name]},
        )
        processed_transition = self.preprocessor._forward(transition)
        batch = transition_to_batch(processed_transition)

        self._coerce_language_batch(batch)
        a = time.perf_counter()
        # Get action
        with torch.no_grad():
            predict_fn = self.policy.predict_action_chunk
            if torch_dynamo is not None:
                predict_fn = torch_dynamo.disable(predict_fn)
            action_chunk = predict_fn(batch)
        b = time.perf_counter()
        print(f"policy run time is {b-a} s")

        if self.postprocessor is not None:
            action_transition = create_transition(action=action_chunk)
            action_transition = self.postprocessor._forward(action_transition)
            action_chunk = action_transition[TransitionKey.ACTION]

        action = action_chunk.squeeze(0).cpu().numpy()
        return action

    def run(self):
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        logging.info(f"Socket server listening on {self.host}:{self.port}")

        try:
            while True:
                # Accept connection
                client_socket, client_address = server_socket.accept()
                logging.info(f"Client connected from {client_address}")

                while True:
                    # Receive message length
                    length_data = self._receive_exact(client_socket, 4)
                    if not length_data:
                        break
                    message_length = struct.unpack('!I', length_data)[0]

                    # Receive message
                    message_data = self._receive_exact(client_socket, message_length)
                    if not message_data:
                        break

                    # Parse observation
                    timestamp, image_data_main, image_data_right, joint_states, task_name = self._parse_observation(message_data)
                    print(f"Task: {task_name}.")
                    print(f"Joint states: {joint_states}")
                    # Process and get action
                    action_chunks = self.process_observation(
                        timestamp, image_data_main, image_data_right, joint_states, task_name
                    )

                    # Send response
                    response = self._serialize_action_chunks(action_chunks)
                    client_socket.send(struct.pack('!I', len(response)))
                    client_socket.send(response)


        except KeyboardInterrupt:
            logging.info("Server shutting down...")
        finally:
            server_socket.close()

    def _receive_exact(self, sock, n):
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _parse_observation(self, data):
        data_offset = 0

        # Read timestamp
        timestamp = struct.unpack('!f', data[data_offset:data_offset+4])[0]
        data_offset += 4

        # Read image length and data
        image_length = struct.unpack('!I', data[data_offset:data_offset+4])[0]
        data_offset += 4
        image_data_main = data[data_offset:data_offset+image_length]
        data_offset += image_length

        # Read right image length and data
        image_right_length = struct.unpack('!I', data[data_offset:data_offset+4])[0]
        data_offset += 4
        image_data_right = data[data_offset:data_offset+image_right_length]
        data_offset += image_right_length

        # Read joint states
        joint_count = struct.unpack('!I', data[data_offset:data_offset+4])[0]
        data_offset += 4
        joint_states = []
        for _ in range(joint_count):
            value = struct.unpack('!f', data[data_offset:data_offset+4])[0]
            joint_states.append(value)
            data_offset += 4
        # Read task name
        task_length = struct.unpack('!I', data[data_offset:data_offset+4])[0]
        data_offset += 4
        task_name = data[data_offset:data_offset+task_length].decode('utf-8')
        data_offset += task_length

        return timestamp, image_data_main, image_data_right, joint_states, task_name

    def _serialize_action_chunks(self, action_chunks):
        if action_chunks is None:
            return struct.pack('!II', 0, 0)
        if action_chunks.ndim == 1:
            action_chunks = action_chunks[None, :]
        if self.action_chunk_size is not None and self.action_chunk_size > 0:
            action_chunks = action_chunks[: self.action_chunk_size]
        n_steps = action_chunks.shape[0]
        n_dim = action_chunks.shape[1] if n_steps > 0 else 0
        if n_steps == 0 or n_dim == 0:
            return struct.pack('!II', 0, 0)
        flattened = action_chunks.flatten().tolist()
        header = struct.pack('!II', n_steps, n_dim)
        payload = struct.pack(f'!{len(flattened)}f', *flattened)
        return header + payload


def main():
    server = PolicyServer()
    server.run()


if __name__ == "__main__":
    main()
