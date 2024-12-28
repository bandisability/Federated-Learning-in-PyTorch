import os
import sys
import time
import torch
import argparse
import traceback
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, RMSprop

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model


def dynamic_client_selection(clients, performance_metrics, top_k=5):
    """Dynamically selects top-performing clients based on performance metrics.

    Args:
        clients: List of available clients.
        performance_metrics: Dictionary of client performance metrics (e.g., accuracy, loss).
        top_k: Number of clients to select for the current round.

    Returns:
        List of selected clients.
    """
    sorted_clients = sorted(performance_metrics.items(), key=lambda x: x[1], reverse=True)
    return [client for client, _ in sorted_clients[:top_k]]


def configure_optimizer(model, optimizer_type="Adam", lr=0.01):
    """Configures the optimizer for local training.

    Args:
        model: The model to optimize.
        optimizer_type: The type of optimizer (default: Adam).
        lr: Learning rate for the optimizer.

    Returns:
        Configured optimizer.
    """
    if optimizer_type == "Adam":
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type == "RMSprop":
        return RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def train_local_model(client_data, model, optimizer, epochs=5):
    """Trains a local model on the given client data.

    Args:
        client_data: DataLoader instance for the client's data.
        model: The model to train.
        optimizer: Optimizer for the model.
        epochs: Number of training epochs.

    Returns:
        Trained model and training loss.
    """
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(client_data):
            data, target = data.to(model.device), target.to(model.device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model, total_loss / len(client_data.dataset)


def main(args, writer):
    """Main program for federated learning with dynamic client selection and advanced optimizations."""
    set_seed(args.seed)

    # Load datasets
    server_dataset, client_datasets = load_dataset(args)

    # Validate arguments
    args = check_args(args)

    # Load model
    model, args = load_model(args)

    # Initialize server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(
        args=args, 
        writer=writer, 
        server_dataset=server_dataset, 
        client_datasets=client_datasets, 
        model=model
    )

    # Federated learning loop
    client_performance = {client_id: 0 for client_id in client_datasets.keys()}  # Initialize performance metrics
    for curr_round in range(1, args.R + 1):
        server.round = curr_round

        # Dynamic client selection
        selected_ids = dynamic_client_selection(client_datasets.keys(), client_performance, top_k=int(args.C * len(client_datasets)))

        # Train selected clients
        for client_id in selected_ids:
            client_data = client_datasets[client_id]
            optimizer = configure_optimizer(server.model, optimizer_type="Adam", lr=args.lr)
            server.model, loss = train_local_model(client_data, server.model, optimizer, epochs=args.E)
            client_performance[client_id] = -loss  # Update performance metric (negative loss as higher is better)

        # Evaluate global model
        if curr_round % args.eval_every == 0 or curr_round == args.R:
            server.evaluate(excluded_ids=selected_ids)
    
    server.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str, help="Experiment name.")
    parser.add_argument('--dataset', required=True, type=str, help="Dataset name.")
    parser.add_argument('--model_name', required=True, type=str, help="Model name.")
    parser.add_argument('--algorithm', required=True, type=str, help="Federated algorithm.")
    parser.add_argument('--R', type=int, default=100, help="Number of rounds.")
    parser.add_argument('--E', type=int, default=5, help="Local epochs.")
    parser.add_argument('--C', type=float, default=0.1, help="Client fraction.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--eval_every', type=int, default=10, help="Evaluation frequency.")
    parser.add_argument('--use_tb', action='store_true', help="Enable TensorBoard.")
    args = parser.parse_args()

    writer = SummaryWriter() if args.use_tb else None
    main(args, writer)

