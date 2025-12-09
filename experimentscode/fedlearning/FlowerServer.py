import logging

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvgM, FedProx
from torch.utils.data import DataLoader

from .my_server_class import MyServer
from .task import *


def _prepare_eval_tensors(dataset, device, img_key="img", target_key="label"):
    """Return tensors for centralized evaluation when FEMNIST arrays are available."""
    images = getattr(dataset, "images", None)
    labels = getattr(dataset, "labels", None)
    if images is None or labels is None:
        return None, None
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    return images_tensor, labels_tensor


def _simple_test(net, images_tensor, labels_tensor):
    """Evaluate using preloaded tensors to avoid DataLoader overhead."""
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    net.eval()
    with torch.no_grad():
        outputs = net(images_tensor)
        loss = criterion(outputs, labels_tensor).item()
        accuracy = (outputs.argmax(dim=1) == labels_tensor).float().mean().item()
    net.train()
    return loss, accuracy


def _loader_test(net, dataset, device, img_key="img", target_key="label"):
    """Fallback evaluation using a simple DataLoader loop."""
    net.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    dataloader = DataLoader(dataset, batch_size=256)
    total_loss, total_examples, correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[img_key].to(device)
            labels = batch[target_key].to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)
    net.train()
    avg_loss = total_loss / max(total_examples, 1)
    accuracy = correct / max(total_examples, 1)
    return avg_loss, accuracy


def get_evaluate_fn(model, dataset_name, device, logger):
    dataset = get_dataset(partition="server_val_data", dataset=dataset_name, dataset_type="val")
    tensors = _prepare_eval_tensors(dataset, device)

    def evaluate(server_round, parameters, config):
        set_weights(model, parameters)
        if all(tensors):
            loss, accuracy = _simple_test(model.to(device), *tensors)
        else:
            loss, accuracy = _loader_test(model.to(device), dataset, device)
        logger.info(f"[Eval] round={server_round} loss={loss:.6f} acc={accuracy:.6f}")
        return loss, {"accuracy": accuracy}

    return evaluate


def get_logger(log_path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{log_path}/server.log')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def server_fn(context: Context):
    dataset = context.run_config["dataset"]
    log_path = context.run_config["log"]
    use_zmq = context.run_config["zmq"]
    onos_address = context.run_config["onos"]
    fl_server_address = context.run_config["fl-server-address"]
    num_rounds = context.run_config["rounds"]
    num_clients = context.run_config["clients"]

    dataset_meta = get_dataset_meta(dataset)
    net = build_model(
        dataset,
        num_classes=dataset_meta["num_classes"],
        in_channels=dataset_meta["in_channels"],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = ServerConfig(num_rounds=num_rounds)

    logger = get_logger(log_path)

    zmq_handler = ZMQHandler(onos_address, fl_server_address) if use_zmq else None
    client_manager = MySimpleClientManager(zmq_handler, logger)

    strategy = FedAvgM(
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_fit=1,
        fraction_evaluate=0,
        server_momentum=0.9,
        evaluate_fn=get_evaluate_fn(net, dataset, device, logger),
        initial_parameters=ndarrays_to_parameters(get_weights(net))
    )

    server = MyServer(client_manager=client_manager, zmq_handler=zmq_handler, strategy=strategy,
                      log_path=log_path, logger=logger)

    return ServerAppComponents(config=config, server=server)


app = ServerApp(server_fn=server_fn)
