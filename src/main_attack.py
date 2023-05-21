from collections import namedtuple
import copy
import hydra
import numpy as np
from omegaconf import DictConfig
import wandb

import methods
import utils
import attack
import data
import eval


# %%
@hydra.main(
    version_base=None,
    config_path=utils.get_project_root() + "/configs",
    config_name="config_attack",
)
def main(config: DictConfig) -> None:
    _results_path, cache_dir = utils.init(config)

    if type(config.methods_args.method_names) == str:
        config.methods_args.method_names = [config.methods_args.method_names]

    # region training function
    methods_dict = methods.init(**config.methods_args)
    method_name, method_dict = list(methods_dict.items())[0]

    # region score function
    # implicit assumption that all models have the same prior and we can score them by ranking them acc. to their log likelihood
    if method_name in ["Chaudhuri_sklearn", "Chaudhuri"]:
        w = 1
    else:
        w = list(method_dict["add_eps_data_fn"].values())[0](
            config.attack.epsilon, 1
        )
    score_fn = methods.create_loglik_fn(
        config.methods_args.task_name,
        config.methods_args.method_names[0],
        config.attack.epsilon,
        w,
    )
    # endregion

    if (
        config.data.group == "adversary"
        and config.methods_args.task_name == "logReg"
    ):
        Ds = namedtuple("Dataset", ["x", "y"])
        if config.methods_args.method_names[0] not in [
            "Chaudhuri",
            "Chaudhuri_sklearn",
        ]:
            ds0 = Ds(
                np.array(((-1,), (0,))),  # x
                np.array((1, -1)),  # y
            )
            d0 = Ds(
                np.array(((-1,),)),
                np.array((1,)),
            )
            d1 = Ds(
                np.array(((1,),)),
                np.array((1,)),
            )
            n_mcmc = 1000

            new_meth_dict = copy.deepcopy(method_dict)
            new_meth_dict["stan_args"]["iter_sampling"] = n_mcmc

            _, _, mcmc_samples = methods.fit(
                method_name,
                new_meth_dict,
                ds0.x,
                ds0.y,
                [config.attack.epsilon],
                seed=0,
                cache_dir=cache_dir,
            )
            norm_const = 0
            for i in range(len(mcmc_samples["theta"])):
                l0 = score_fn(d0, {"theta": mcmc_samples["theta"][i : i + 1]})
                l1 = score_fn(d1, {"theta": mcmc_samples["theta"][i : i + 1]})
                norm_const += np.exp(l1 - l0)
            norm_const = norm_const / n_mcmc
            # norm_const = 1
        else:
            norm_const = 1
    else:
        raise NotImplementedError("Norm const not implemented for this case.")
        norm_const = 1

    if config.attack.private:

        def train_fn(data, idx_round):
            data_X, data_y = data.x, data.y
            _theta_hat, theta_hat_priv, _ori_samples = methods.fit(
                method_name,
                method_dict,
                data_X,
                data_y,
                [config.attack.epsilon],
                seed=config.seed + idx_round,
                cache_dir=cache_dir,
            )
            return theta_hat_priv[list(theta_hat_priv.keys())[0]]

    else:

        def train_fn(data, idx_round):
            data_X, data_y = data.x, data.y
            theta_hat, _theta_hat_priv, _ori_samples = methods.fit(
                method_name,
                method_dict,
                data_X,
                data_y,
                [config.attack.epsilon],
                seed=config.seed + idx_round,
                cache_dir=cache_dir,
            )
            return theta_hat[list(theta_hat.keys())[0]]

    # endregion

    # region craft function
    if config.attack.craft_name == "drop":

        def craft_fn(data):
            data_X, data_y = data
            data_X0 = data_X[config.attack.n_drop :]
            data_y0 = data_y[config.attack.n_drop :]
            data_X1 = data_X[: -config.attack.n_drop]
            data_y1 = data_y[: -config.attack.n_drop]
            return ((data_X0, data_y0), (data_X1, data_y1))

    elif config.attack.craft_name == "flip":

        def craft_fn(data):
            data_X, data_y = data
            data_y1 = data_y.copy()
            if config.methods_args.task_name == "logReg":
                data_y1[: config.attack.n_drop] = -data_y[
                    : config.attack.n_drop
                ]
            elif config.methods_args.task_name in ["linReg", "mlpReg"]:
                data_y1[: config.attack.n_drop] = (
                    -np.sign(data_y[: config.attack.n_drop] - data_y.mean())
                    * data_y.std()
                    * 1000
                )

            return ((data_X, data_y), (data_X, data_y1))

    elif config.attack.craft_name == "blank":

        def craft_fn(data):
            data_X, data_y = data
            data_y1 = data_y.copy()
            data_X1 = data_X.copy()
            if config.methods_args.task_name == "logReg":
                data_X1[: config.attack.n_drop] = 0
                data_y1[: config.attack.n_drop] = np.mean(data_y) < 0.5
            elif config.methods_args.task_name in ["linReg", "mlpReg"]:
                data_X1[: config.attack.n_drop, data_X1.shape[1] > 1 :] = 0
                data_y1[: config.attack.n_drop] = (
                    -np.sign(data_y.mean()) * data_y.std() * 1000
                )

            return ((data_X, data_y), (data_X1, data_y1))

    if config.data.group == "simulated":

        def train_dataset_fn(idx_round):
            ds = data.get_dataset(
                config,
                seed=config.seed + idx_round,
                n_feats=config.attack.n_feats,
                n_obs=config.attack.n_obs + config.attack.n_drop,
                eval=True,
            )
            # utils.reg_plot(ds)

            return ds

    elif config.data.group == "uci":
        train_dataset = data.get_dataset(
            config,
            seed=config.seed,
            n_feats="None",
            eval=True,
        )

        def train_dataset_fn(_idx_round):
            ind_obs = np.random.choice(
                train_dataset.x.shape[0],
                config.attack.n_obs + config.attack.n_drop,
                replace=False,
            )
            if (
                config.attack.n_feats is not None
                and config.attack.n_feats != "None"
            ):
                ind_feats = np.random.choice(
                    train_dataset.x.shape[1],
                    config.attack.n_feats,
                    replace=False,
                )
                data_X = train_dataset.x[ind_obs][:, ind_feats]
            else:
                data_X = train_dataset.x[ind_obs]

            data_y = train_dataset.y[ind_obs]
            new_ds = copy.deepcopy(train_dataset)
            new_ds.x = data_X
            new_ds.y = data_y
            return new_ds

    if config.data.group == "adversary":
        config.attack.n_feats = 1
        config.attack.n_obs = 2
        config.attack.n_drop = 1
        Ds = namedtuple("Dataset", ["x", "y"])

        if config.methods_args.task_name == "logReg":

            def wrap_craft_fn(idx_round):
                ds0 = Ds(
                    np.array(((-1,), (0,))),
                    np.array((1, -1)),
                )
                ds1 = Ds(
                    np.array(((1,), (0,))),
                    np.array((1, -1)),
                )
                return ds0, ds1, None

        elif config.methods_args.task_name in ["linReg", "mlpReg"]:

            def train_dataset_fn(_idx_round):
                ds0 = Ds(
                    np.array(((0,), (1e-7,))),
                    np.array((1e7,), (-1e-7,)),
                )
                ds1 = Ds(
                    np.array(((0,), (-1e-7,))),
                    np.array((1e7,), (-1e-7,)),
                )
                return ds0, ds1, None

    else:

        def wrap_craft_fn(idx_round):
            # read real data
            ds = train_dataset_fn(idx_round)
            j = 0
            while (
                config.methods_args.task_name == "logReg"
                and sum(ds.y != ds.y[0]) == 0
            ):
                j += 1
                ds = train_dataset_fn(idx_round + config.attack.n_rounds + j)
            ds0 = utils.ObjectFromDict({})
            ds1 = utils.ObjectFromDict({})
            # ds0 = utils.ObjectFromDict({"theta_gen": ds.theta_gen})
            # ds1 = utils.ObjectFromDict({"theta_gen": ds.theta_gen})
            (ds0.x, ds0.y), (ds1.x, ds1.y) = craft_fn((ds.x, ds.y))

            return ds0, ds1, ds

    # endregion

    # region eval fn
    predict_fn = eval.create_predict_fn(
        config.methods_args.task_name, method_name
    )
    if config.methods_args.task_name in ["linReg", "mlpReg"]:
        metrics_fn = eval.rmse_loss
        metrics_name = "rmse"
    else:
        metrics_fn = eval.roc_auc
        metrics_name = "auc"

    if config.data.group == "adversary":
        eval_fn = None
    else:

        def eval_fn(trained, ds):
            pred = predict_fn(ds.x_eval, trained)
            return metrics_fn(pred, ds.y_eval)

    # endregion

    eps_low, fpr, fnr, acc, eval_mean, eval_std = attack.lower_bound_audit_nasr(
        train_fn,
        score_fn,
        wrap_craft_fn,
        config.attack.n_rounds,
        config.methods_args.delta,
        eval_fn,
        norm_const,
    )

    wandb.log(
        {
            "eps_low": eps_low,
            "fpr": fpr,
            "fnr": fnr,
            "acc": acc,
            "acc_se": np.sqrt(acc * (1 - acc) / config.attack.n_rounds),
            f"{metrics_name}_eval_mean": eval_mean,
            f"{metrics_name}_eval_std": eval_std,
        }
    )

    log_results = [
        config.run_name.prefix,
        config.seed,
        config.methods_args.task_name,
        config.methods_args.method_names[0],
        *list(config.attack.values()),
        eps_low,
        fpr,
        fnr,
        acc,
        eval_mean,
        eval_std,
        metrics_name,
    ]
    log_results = [
        np.round(i, 4) if type(i) == np.float64 else i for i in log_results
    ]
    with open("results/attack.csv", "a") as f:
        f.write(",".join([str(i) for i in log_results]) + "\n")

    utils.finish(cache_dir)


if __name__ == "__main__":
    main()
