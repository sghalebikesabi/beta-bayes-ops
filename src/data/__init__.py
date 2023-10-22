import numpy as np

from .simulated import SimulatedDataset
from .uci import UCIDataset

DATASET_DICT = {
    "simulated": SimulatedDataset,
    "uci": UCIDataset,
}


def get_dataset(config, seed, n_feats, n_obs=None, eval=True):
    if n_obs is None and "None" in config.sweeps.n_obs:
        n_obs = "None"
    elif n_obs is None:
        n_obs = max(config.sweeps.n_obs)

    ds = DATASET_DICT[config.data.group](
        task_name=config.methods_args.task_name,
        seed=seed,
        n_samples=n_obs,
        n_samples_eval=eval * 5000,
        n_feats=n_feats,
        add_constant=config.methods_args.task_name != "mlpReg"
        and (n_feats == "None" or n_feats > 1),
        hidden_size=config.methods_args.get("hidden_size", None),
        name=config.data.name,
        eval_split=config.data.eval_split,
    )

    if config.methods_args.task_name == "logReg":
        assert ds.y.min() == -1 or np.unique(ds.y).size == 1
        assert ds.y.max() == 1 or np.unique(ds.y).size == 1

    return ds
