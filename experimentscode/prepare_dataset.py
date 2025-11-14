from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner,DirichletPartitioner

NUM_CLIENTS = 100
DATASET = 'cifar10'

partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS,partition_by="label", alpha=0.5,min_partition_size=0.0002)
fds = FederatedDataset(dataset=DATASET, partitioners={"train": partitioner, "test": 1})
for partition_id in range(NUM_CLIENTS):
    partition = fds.load_partition(partition_id, "train")
    partition.save_to_disk(f"data/{DATASET}/client_{partition_id}_data")
testset = fds.load_split("test")
testset.save_to_disk(f"data/{DATASET}/server_val_data")
