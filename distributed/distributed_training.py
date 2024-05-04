import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
import pandas as pd

def setup(rank: int, world_size: int):
    """Initialize the distributed environment."""
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for GPU training (use gloo for CPU)
        init_method="tcp://localhost:12355",  # Address for all nodes
        world_size=world_size,
        rank=rank
    )
    torch.manual_seed(42)


def cleanup():
    """Finalize the distributed environment."""
    dist.destroy_process_group()


def train(rank: int, world_size: int, model, data: pd.DataFrame):
    """Training function executed in each process."""
    setup(rank, world_size)

    # Split the data into train-test sets
    feature_columns = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "price_pct_change", "on_balance_volume"]
    target_column = "close"
    X = data[feature_columns].fillna(0)
    y = data[target_column].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DDP model
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    # Create distributed data sampler and loader
    train_sampler = DistributedSampler(X_train, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(X_train, sampler=train_sampler, batch_size=32)

    # Training loop
    criterion = torch.nn.MSELoss().to(rank)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs = batch.to(rank)
            labels = y_train[batch.index].to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    cleanup()


def run_distributed_training(data: pd.DataFrame, model):
    """Main entry point for distributed training."""
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs available for distributed training.")
    
    torch.multiprocessing.spawn(train, args=(world_size, model, data), nprocs=world_size, join=True)
