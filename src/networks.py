"""
Neural network building blocks for entropy-regularised Bermudan option pricing.
"""

import torch
import torch.nn as nn


class ContinuationNet(nn.Module):
    """
    Fully connected network for approximating continuation values.

    Input : state X_t  (d-dimensional)
    Output: scalar continuation value script_V(X)

    Architecture
    ------------
    Linear(d, width)
      -> [Tanh -> Linear(width, width)] × (depth - 2)
      -> Tanh -> Linear(width, 1)

    Parameters
    ----------
    input_dim : int
        Dimensionality d of the input state.
    hidden_width : int
        Number of neurons in every hidden layer.
    depth : int
        Total number of linear layers (minimum 2).
    """

    def __init__(self, input_dim: int, hidden_width: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_width)]
        for _ in range(depth - 2):
            layers.extend([nn.Tanh(), nn.Linear(hidden_width, hidden_width)])
        layers.extend([nn.Tanh(), nn.Linear(hidden_width, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, d)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        return self.net(x).squeeze(-1)


def train_network(
    net: nn.Module,
    X_inputs: torch.Tensor,
    y_targets: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 8192,
    device: str = 'cpu',
    verbose: bool = False,
) -> list[float]:
    """
    Train *net* with MSE loss using the Adam optimiser.

    Parameters
    ----------
    net : nn.Module
        Network to train (modified in-place).
    X_inputs : torch.Tensor, shape (n_paths, d)
        Input features.
    y_targets : torch.Tensor, shape (n_paths,)
        Regression targets.
    n_epochs : int
        Number of full passes through the training data.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    device : str
        Torch device string ('cpu', 'cuda', …).
    verbose : bool
        If True, print epoch loss every 100 epochs.

    Returns
    -------
    list[float]
        Per-epoch mean MSE loss.
    """
    net = net.to(device)
    if not isinstance(X_inputs, torch.Tensor):
        X_inputs = torch.tensor(X_inputs, dtype=torch.float32)
    if not isinstance(y_targets, torch.Tensor):
        y_targets = torch.tensor(y_targets, dtype=torch.float32)

    X_inputs = X_inputs.to(device).float()
    # Ensure targets are 1-D to prevent MSELoss from broadcasting into an
    # (n, n) matrix when targets arrive with shape (n, 1).
    y_targets = y_targets.to(device).float().squeeze()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_inputs.shape[0]
    loss_history: list[float] = []

    net.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        X_shuf = X_inputs[perm]
        y_shuf = y_targets[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            X_batch = X_shuf[start : start + batch_size]
            y_batch = y_shuf[start : start + batch_size]

            optimizer.zero_grad()
            pred = net(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  loss={epoch_loss:.6f}")

    net.eval()
    return loss_history
