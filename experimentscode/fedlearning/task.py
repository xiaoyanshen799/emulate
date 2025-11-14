import concurrent.futures
import threading
import time
from collections import OrderedDict
from enum import Enum
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from datasets import Dataset as HFDataset
from flwr.common import GetPropertiesIns
from flwr.server import SimpleClientManager
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from torchvision.models import mobilenet_v2


def convert_bn_to_gn(model, num_groups=8):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            # Replace with GroupNorm - note the different parameters
            num_channels = module.num_features
            setattr(model, name, torch.nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels
            ))
        else:
            convert_bn_to_gn(module, num_groups)
    return model


class CifarCnnModel(torch.nn.Module):
    """Baseline CNN for CIFAR-10-like RGB inputs."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(input_tensor))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MobileNetModel(torch.nn.Module):
    """MobileNetV2 backbone adapted for grayscale FEMNIST-like inputs."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        alpha: float = 1.0,
        min_hw: int = 96,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.min_hw = min_hw
        self.replicate_grayscale = in_channels == 1
        if in_channels == 3 or self.replicate_grayscale:
            self.channel_project = None
        else:
            self.channel_project = torch.nn.Conv2d(in_channels, 3, kernel_size=1)

        self.backbone = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=alpha)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.min_hw is not None:
            if input_tensor.shape[-1] != self.min_hw or input_tensor.shape[-2] != self.min_hw:
                input_tensor = F.interpolate(
                    input_tensor,
                    size=(self.min_hw, self.min_hw),
                    mode="bilinear",
                    align_corners=False,
                )

        if self.replicate_grayscale:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
        elif self.channel_project is not None:
            input_tensor = self.channel_project(input_tensor)

        return self.backbone(input_tensor)


def _resolve_data_root() -> Path:
    candidates = [
        os.environ.get("FL_DATA_ROOT"),
        "/app/data",
        Path(__file__).resolve().parent.parent / "data",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[-1])


_DATA_ROOT = _resolve_data_root()
_FEMNIST_DIR = _DATA_ROOT / "femnist"
_FEMNIST_MIN_CLASSES = 62
_DATASET_ARCH = {"femnist": "mobilenet", "cifar10": "cifar_cnn"}


class FEMNISTDataset(TorchDataset):
    """Simple torch Dataset wrapper returning dict batches."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"img": image, "label": label}


def _prepare_images(images: np.ndarray, name: str) -> np.ndarray:
    if images.ndim == 4:
        processed = images
    elif images.ndim == 3:
        processed = images[..., None]
    elif images.ndim == 2:
        side = int(np.sqrt(images.shape[1]))
        if side * side != images.shape[1]:
            raise ValueError(f"Cannot reshape {name} of shape {images.shape} into square images.")
        processed = images.reshape((images.shape[0], side, side, 1))
    else:
        raise ValueError(f"Expected {name} to be 2-D, 3-D, or 4-D, got {images.shape}.")

    processed = processed.astype(np.float32, copy=False)
    if processed.size and processed.max() > 1.0:
        processed /= 255.0
    return processed


def _load_npz_split(path: Path, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"FEMNIST split not found: {path}")
    with np.load(path) as npz:
        return npz[x_key].astype(np.float32), npz[y_key].astype(np.int64)


def _split_train_val(images: np.ndarray, labels: np.ndarray, split_ratio: float = 0.9) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    split_idx = int(split_ratio * len(images))
    split_idx = max(1, min(split_idx, len(images) - 1))
    return (images[:split_idx], labels[:split_idx]), (images[split_idx:], labels[split_idx:])


def _femnist_partition(partition: str, dataset_type: str) -> FEMNISTDataset:
    dataset_type = dataset_type.lower()
    clean_partition = "".join(filter(str.isdigit, partition)) or partition
    if partition == "server_val_data" or dataset_type == "test":
        images, labels = _load_npz_split(_FEMNIST_DIR / "test_femnist.npz", "x_test", "y_test")
        images = _prepare_images(images, "x_test")
        return FEMNISTDataset(images, labels)

    images, labels = _load_npz_split(_FEMNIST_DIR / f"{clean_partition}.npz", "x_train", "y_train")
    images = _prepare_images(images, f"x_train_{clean_partition}")
    (train_images, train_labels), (val_images, val_labels) = _split_train_val(images, labels)

    if dataset_type in {"val", "validation"}:
        return FEMNISTDataset(val_images, val_labels)
    return FEMNISTDataset(train_images, train_labels)


def get_dataset(partition, dataset="cifar10", img_key="img", dataset_type="train"):
    dataset = dataset.lower()
    dataset_type = dataset_type.lower()
    if dataset == "femnist":
        return _femnist_partition(partition, dataset_type)

    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def apply_transforms(batch):
        pytorch_transforms = train_transformer if dataset_type == "train" else test_transformer
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    partition_path = _DATA_ROOT / dataset / partition
    partition_ds = HFDataset.load_from_disk(str(partition_path))
    return partition_ds.with_transform(apply_transforms)


def get_dataset_meta(dataset: str) -> Dict[str, int]:
    dataset = dataset.lower()
    if dataset == "femnist":
        return {"in_channels": 1, "num_classes": _FEMNIST_MIN_CLASSES, "model": "mobilenet"}
    return {"in_channels": 3, "num_classes": 10, "model": "cifar_cnn"}


def build_model(dataset: str, *, num_classes: int, in_channels: int, alpha: float = 1.0) -> torch.nn.Module:
    arch = _DATASET_ARCH.get(dataset.lower(), "mobilenet")
    if arch == "cifar_cnn":
        return CifarCnnModel(num_classes=num_classes, in_channels=in_channels)
    return MobileNetModel(in_channels=in_channels, num_classes=num_classes, alpha=alpha)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class ZMQHandler:
    class MessageType(Enum):
        UPDATE_DIRECTORY = 1
        SERVER_TO_CLIENTS = 2
        CLIENT_TO_SERVER = 3

    def __init__(self, onos_server, fl_server_address):
        self.onos_server_address = onos_server
        self.fl_server_address = fl_server_address
        self.snd_socket = None
        self.recv_socket = None

        self.init_zmq()
        threading.Thread(target=self.zmq_bridge, args=(self.snd_socket, self.recv_socket,),
                         daemon=True).start()

    def init_zmq(self):
        context = zmq.Context()
        self.recv_socket = context.socket(zmq.PULL)
        self.recv_socket.bind(f"tcp://{self.fl_server_address}:5555")
        self.snd_socket = context.socket(zmq.PUSH)
        self.snd_socket.connect(f"tcp://{self.onos_server_address}:5555")

    def send_data_to_server(self, message_type: MessageType, message):
        model_update = {"sender_id": "server", "message_type": message_type.value, "message": message,
                        "time_ms": round(time.time() * 1000)}
        self.snd_socket.send_json(model_update)

    @staticmethod
    def zmq_bridge(snd_socket, recv_socket):
        while True:
            snd_socket.send(recv_socket.recv())


class MySimpleClientManager(SimpleClientManager):
    def __init__(self, zmq_handler, logger) -> None:
        super().__init__()
        self.clients_info = {}
        self.zmq_handler = zmq_handler
        self.logger = logger

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        sampled_clients = super().sample(num_clients, min_num_clients, criterion)
        new_clients = list(filter(lambda c: c.cid not in self.clients_info, sampled_clients))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            submitted_fs = {
                executor.submit(self.get_props, client_proxy) for client_proxy in new_clients
            }

            concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )
        return sampled_clients

    def unregister(self, client):
        self.logger.info(f"Unregistering {client.cid}")
        super().unregister(client)

    def get_props(self, client):
        try:
            tik = time.time()
            properties = client.get_properties(GetPropertiesIns({}), None, None).properties
            self.logger.info(f"Getting Properties for Client {client.cid} took {time.time() - tik}s")
            if properties:
                self.clients_info[client.cid] = dict(properties)
                if self.zmq_handler:
                    self.logger.info(f"Sending this data to server {properties}", )
                    self.zmq_handler.send_data_to_server(ZMQHandler.MessageType.UPDATE_DIRECTORY, dict(properties))
        except Exception as e:
            self.logger.error(e)
