import hydra
import numpy as np
from omegaconf import DictConfig
import os
import wandb

import data
import eval
import methods
import utils


@hydra.main(
    version_base=None,
    config_path=utils.get_project_root() + "/configs",
    config_name="config_sweep",
)
def main(config: DictConfig) -> None:
    results_path, cache_dir = utils.init(config)

    # region ---------------------------------------------- estimation
    methods_dict = methods.init(**config.methods_args)

    eval_metrics = {}
    eval_metrics_priv = {}

    for j in range(config.sweeps.n_rep):
        for d in config.sweeps.n_feats:
            theta_hat = {}
            theta_hat_priv = {}

            dataset = data.get_dataset(
                config, seed=config.seed + j, n_feats=d, eval=True
            )
            if d == "None":
                d = dataset.x.shape[1]

            config.sweeps.n_obs = [
                n
                if (n != "None" and n <= dataset.x.shape[0])
                else dataset.x.shape[0]
                for n in config.sweeps.n_obs
            ]

            for n in config.sweeps.n_obs:
                print(f"++++++++running {j}-{d}-{n} iteration++++++++")

                data_y = 0
                while len(np.unique(data_y)) < 2:
                    ind = np.random.choice(
                        dataset.x.shape[0],
                        n,
                        replace=False,
                    )
                    data_X, data_y = dataset.x[ind], dataset.y[ind]

                for method_name, method_dict in methods_dict.items():
                    print(f"--------running {method_name}--------")
                    theta_hat_nj, theta_hat_priv_nj, _ori_params = methods.fit(
                        method_name=method_name,
                        method_dict=method_dict,
                        data_X=data_X,
                        data_y=data_y,
                        epsilon_vec=config.sweeps.epsilon,
                        seed=config.seed + j,
                        cache_dir=cache_dir,
                    )
                    for key, val_priv in theta_hat_priv_nj.items():
                        k, e = key.split("-")
                        theta_hat[f"{k}-{j}-{d}-{n}-{e}"] = theta_hat_nj[key]
                        theta_hat_priv[f"{k}-{j}-{d}-{n}-{e}"] = val_priv
                # endregion

            for sweep_name in theta_hat.keys():
                try:
                    eval_metrics_j_d, eval_metrics_priv_j_d = eval.eval(
                        task_name=config.methods_args.task_name,
                        method_name=sweep_name.split("-")[0],
                        theta_hat=theta_hat[sweep_name],
                        theta_hat_priv=theta_hat_priv[sweep_name],
                        eval_dataset=dataset,
                        suffix="-" + sweep_name,
                    )
                except Exception as e:
                    print(sweep_name)
                    raise e
                eval_metrics.update(eval_metrics_j_d)
                eval_metrics_priv.update(eval_metrics_priv_j_d)

    # region ---------------------------------------------- logging
    eval_df = eval.create_eval_df(
        eval_metrics,
        eval_metrics_priv,
    )
    if config.logging.final_params_plot:
        eval_df.to_csv(results_path + "/eval_df.csv", index=False)
    wandb.log({"eval_df": wandb.Table(dataframe=eval_df)})
    print(eval_df)

    if config.logging.final_predictions_save:
        utils.sweep_plot(
            name=results_path + "/eval",
            eval_df=eval_df,
            **config.logging.plot_args,
        )

    # endregion

    utils.finish(cache_dir)


if __name__ == "__main__":
    main()
