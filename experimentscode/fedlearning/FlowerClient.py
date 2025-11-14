import gc
import json
import logging
import random
import socket
import timeit

import flwr as fl
import psutil
import torch.optim.lr_scheduler
from flwr.client import ClientApp
from flwr.common import Context
from torch.utils.data import DataLoader

from .task import *

torch.set_num_threads(1)
flwr_client = None


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, **kwargs):
        self.cid = kwargs.pop("cid")
        self.node_id = kwargs.pop("node_id")
        self.zmq_socket = kwargs.pop("zmq_socket")
        self.local_epochs = kwargs.pop("local_epochs")
        self.batch_size = kwargs.pop("batch_size")
        self.logger = kwargs.pop("logger")
        dataset = kwargs.pop("dataset").lower()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_meta = get_dataset_meta(dataset)
        self.net = build_model(
            dataset,
            num_classes=dataset_meta["num_classes"],
            in_channels=dataset_meta["in_channels"],
        ).to(device=self.device)

        partition = str(self.cid) if dataset == "femnist" else f"client_{self.cid}_data"
        self.train_set = get_dataset(dataset=dataset, partition=partition, dataset_type="train")
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=3, min_lr=0.0005, threshold=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.logger.info("FlowerClient inside __init__")

    def fit(self, parameters, config):
        try:
            self.logger.info(f"Memory usage before Fit(): {psutil.Process().memory_info().rss / 1024 ** 2} MB")
            self.logger.info(f"LR Value: {self.scheduler.get_last_lr()}")
            computing_start_time = timeit.default_timer()

            set_weights(self.net, parameters)

            loss = self.train()

            computing_finish_time = timeit.default_timer()

            metrics = {"client": self.cid, "computing_start_time": computing_start_time,
                       "computing_finish_time": computing_finish_time, "loss": loss}

            gc.collect()
            return get_weights(self.net), len(self.train_loader.dataset), metrics
        except Exception as e:
            self.logger.error(f"ERROR in FIT {e}")
            raise e

    def train(self):
        self.net.train()
        total_steps = len(self.train_loader)
        running_loss = 0.0
        for epoch in range(self.local_epochs):
            for i, batch in enumerate(self.train_loader):
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(images), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    self.logger.info(f"Step {i} Epoch {epoch}")
                # update the controller after 80% of the steps are done
                if self.zmq_socket and epoch == (self.local_epochs - 1) and i == round(total_steps * 0.85):
                    self.send_data_to_server("client_to_server_path", ZMQHandler.MessageType.CLIENT_TO_SERVER.value)

        avg_loss = running_loss / len(self.train_loader)
        self.scheduler.step(avg_loss)
        return avg_loss

    def get_properties(self, config):
        try:
            interfaces = [name for name in psutil.net_if_addrs().keys() if name != "lo"]
            interface = next(
                (name for name in interfaces if name.startswith(("flclient", "fc"))),
                interfaces[0] if interfaces else "",
            )
            addrs = psutil.net_if_addrs().get(interface, "")
            ip_address = next((addr.address for addr in addrs if addr.family == socket.AF_INET), "0.0.0.0")
            mac_address = next((addr.address for addr in addrs if addr.family == socket.AF_PACKET), "00:00:00:00:00:00")
            return {"client_cid": str(self.cid), "node_id": str(self.node_id), "ip": ip_address, "mac": mac_address}
        except Exception as e:
            self.logger.error(f"in get_properties {e}")

    def send_data_to_server(self, data, message_type):
        model_update = {"sender_id": self.cid, "message_type": message_type, "message": data,
                        "time_ms": round(time.time() * 1000)}
        message = json.dumps(model_update)
        self.zmq_socket.send_string(message)
        self.logger.info(f"Sent message: {message}")


def init_zmq(fl_server, logger):
    logger.info("Initializing ZMQ")
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect(f"tcp://{fl_server}:5555")
    return zmq_socket


def get_logger(cid, log_path):
    logger = logging.getLogger(f'{cid}_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{log_path}/client_{cid}.log')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def init(context: Context):
    cid = context.node_config.get("cid", random.randint(0, 30))
    node_id = context.node_id
    dataset = context.run_config["dataset"]
    log_path = context.run_config["log"]
    use_zmq = context.run_config["zmq"]
    fl_server_address = context.run_config["fl-server-address"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]

    logger = get_logger(cid, log_path)

    zmq_socket = init_zmq(fl_server_address, logger) if use_zmq else None

    return FlowerClient(cid=cid, node_id=node_id, dataset=dataset, zmq_socket=zmq_socket, local_epochs=local_epochs,
                        batch_size=batch_size, logger=logger)


def client_fn(context: Context):
    global flwr_client
    if flwr_client is None:
        t = time.strftime("%H:%M:%S")
        flwr_client = (init(context), t)
    return flwr_client[0].to_client()


app = ClientApp(client_fn, )
