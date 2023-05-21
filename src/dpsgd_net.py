import warnings

warnings.simplefilter("ignore")

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm
import torchbnn as bnn
from pyvacy import optim as dpoptim
import opacus

import data
import utils

MAX_PHYSICAL_BATCH_SIZE = 5128


def predict(data_X, model):
    model.eval()
    return model(torch.tensor(data_X).float()).detach().numpy().reshape(-1)


class MLP(nn.Module):
    def __init__(self, n_feats, n_hidden, n_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_features=n_feats,
                out_features=n_hidden,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=n_hidden,
                out_features=n_classes,
            ),
        )

    def forward(self, x):
        return self.net(x)


class BayesMLP(nn.Module):
    def __init__(self, n_feats, n_hidden, n_classes):
        super().__init__()

        self.net = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=n_feats,
                out_features=n_hidden,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=n_hidden,
                out_features=n_classes,
            ),
        )

    def forward(self, x):
        return self.net(x)


def dp_train(model, train_loader, optimizer, epoch, device, criterion):
    model.train()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target, model)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                print(
                    f"\tTrain Epoch: {epoch} \t" f"Loss: {np.mean(losses):.6f} "
                )


def pub_train(model, train_loader, optimizer, epoch, device, criterion):
    model.train()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target, model)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print(f"\tTrain Epoch: {epoch} \t" f"Loss: {np.mean(losses):.6f} ")


def create_criterion(run_bnn):
    if run_bnn:
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        kl_weight = 0.01

        def bnn_criterion(output, target, model):
            mse = mse_loss(output, target)
            kl = kl_loss(model)
            cost = mse + kl_weight * kl

            return cost

        return bnn_criterion

    else:
        loss = nn.MSELoss()
        return lambda output, target, model: loss(output, target)


def dpsgd_net(
    data,
    lr,
    clip_norm,
    epochs,
    delta,
    hidden_size,
    batch_size,
    run_bnn,
    seed,
    output_dir=None,
):
    del output_dir

    utils.make_deterministic(seed)
    models = {}

    x, y = torch.tensor(data["X"]).float(), torch.tensor(
        data["y"]
    ).float().reshape((-1, 1))
    p, eps = data["p"], data["eps"]
    epochs = 15 + int(eps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_bnn:
        network = BayesMLP
    else:
        network = MLP
    criterion = create_criterion(run_bnn)

    for dp in [True, False]:
        train_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
        )

        model = network(
            n_feats=p,
            n_hidden=hidden_size,
            n_classes=1,
        ).to(device)

        if dp:
            model = ModuleValidator.fix(model)
            ModuleValidator.validate(model, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        if dp:
            # opacus seems to provide more funcitonality than pyvacy. But it does not work with torchbnn. So we will use pyvacy for BNNs, and opacus otherwise.
            if run_bnn:
                training_parameters = {
                    "l2_norm_clip": clip_norm,
                    "noise_multiplier": opacus.accountants.utils.get_noise_multiplier(
                        target_epsilon=eps,
                        target_delta=delta,
                        sample_rate=x.shape[1] / batch_size,
                        epochs=epochs,
                    ),
                    "lr": lr,
                    "batch_size": batch_size,
                }

                optimizer = dpoptim.DPAdam(
                    params=model.parameters(), **training_parameters
                )
                train = pub_train
            else:
                privacy_engine = PrivacyEngine()

                (
                    model,
                    optimizer,
                    train_loader,
                ) = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    epochs=epochs,
                    target_epsilon=eps,
                    target_delta=delta,
                    max_grad_norm=clip_norm,
                )
                print(
                    f"Using sigma={optimizer.noise_multiplier} and C={clip_norm}"
                )
                train = dp_train

        else:
            train = pub_train

        for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
            train(
                model,
                train_loader,
                optimizer,
                epoch + 1,
                device,
                criterion=criterion,
            )
            with torch.no_grad():
                loss = criterion(model(x), y, model)
                print(f"Train Epoch: {epoch}\tLoss: {loss.item()}")

        prefix = "priv" if dp else "nonpriv"
        models[prefix] = model

    return models


@hydra.main(
    version_base=None,
    config_path=utils.get_project_root() + "/configs",
    config_name="config_sweep",
)
def main(config: DictConfig) -> None:
    d = 10
    train_dataset = data.get_dataset(
        config, seed=config.seed, n_feats=d, eval=True
    )

    for j in range(config.sweeps.n_rep):
        for d in config.sweeps.n_feats:
            train_dataset = data.get_dataset(
                config, seed=config.seed + j, n_feats=d, eval=True
            )

            for n in config.sweeps.n_obs:
                ind = np.random.choice(
                    train_dataset.x.shape[0],
                    n,
                    replace=False,
                )
                data_X, data_y = train_dataset.x[ind], train_dataset.y[ind]
                for eps in config.sweeps.epsilon:
                    dpsgd_net(
                        data=dict(
                            X=data_X,
                            y=data_y,
                            eps=eps,
                            p=d,
                        ),
                        lr=config.methods_args.lr,
                        clip_norm=config.methods_args.clip_norm,
                        epochs=config.methods_args.epochs,
                        delta=config.methods_args.delta,
                        hidden_size=config.methods_args.hidden_size,
                        batch_size=config.methods_args.batch_size,
                        run_bnn=config.methods_args.run_bnn,
                        seed=config.seed,
                        output_dir=None,
                    )


if __name__ == "__main__":
    main()
