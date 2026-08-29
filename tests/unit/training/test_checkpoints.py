from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def test_checkpoint_restores_model_optimizer_epoch_and_rng(tmp_path) -> None:
    from prooflens.training.checkpoints import CheckpointManager

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    manager = CheckpointManager(tmp_path)
    saved_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    path = manager.save(
        "epoch-1",
        model,
        optimizer,
        epoch=1,
        global_step=7,
        config_hash="abc",
    )
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(1)
    with torch.no_grad():
        model.weight.fill_(99)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restored = manager.load(path, model, optimizer)

    assert restored.epoch == 1
    assert restored.global_step == 7
    assert restored.config_hash == "abc"
    assert all(torch.equal(model.state_dict()[name], value) for name, value in saved_weights.items())
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    assert not list(tmp_path.glob("*.tmp"))


def test_mark_best_publishes_complete_checkpoint_atomically(tmp_path) -> None:
    from prooflens.training.checkpoints import CheckpointManager

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save("epoch-1", model, optimizer, 1, 1, "cfg")

    best = manager.mark_best(checkpoint)

    assert best == tmp_path / "best.pt"
    assert best.read_bytes() == checkpoint.read_bytes()
    assert not list(tmp_path.glob("*.tmp"))
