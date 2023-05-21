from cmdstanpy import CmdStanModel
import collections
import copy
import numpy as np
import functools
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor

import eval
import utils


def init(
    task_name,
    method_names,
    stan_dir,
    delta,
    n_warmup,
    n_mcmc,
    hidden_size=None,
    lr=None,
    clip_norm=None,
    epochs=None,
    batch_size=None,
    run_bnn=None,
    prior_mean=0,
    prior_var=9,
    reg_ig_prior_shape=1,
    reg_ig_prior_scale=1,
    beta_w=1,  # always set to 1
    reg=1,
):
    # Loading and compiling stan files
    if task_name == "logReg":
        if "BetaBayes" in method_names:
            betaBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/betaBayes_logisticRegression_ML.stan"
            )
        if "Chaudhuri" in method_names or "Minami" in method_names:
            wKLBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/KLBayes_power_logisticRegression_ML.stan"
            )
        all_methods_default_data = dict(
            mu_0=prior_mean,
            v_0=prior_var,
        )
    elif task_name == "linReg":
        if "BetaBayes" in method_names:
            betaBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/betaBayesnorm_linearmodel.stan"
            )
        if "Chaudhuri" in method_names or "Minami" in method_names:
            wKLBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/KLBayesnorm_power_linearmodel.stan"
            )
        all_methods_default_data = dict(
            mu_0=prior_mean,
            v_0=prior_var,
            a_0=reg_ig_prior_shape,
            b_0=reg_ig_prior_scale,
        )
    elif task_name == "mlpReg":
        if "BetaBayes" in method_names:
            betaBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/betaBayesnorm_OneLayerNeuralNetwork.stan"
            )
        if "Chaudhuri" in method_names or "Minami" in method_names:
            wKLBayes_stan = CmdStanModel(
                stan_file=utils.get_project_root()
                + "/"
                + stan_dir
                + "/KLBayesnorm_power_OneLayerNeuralNetwork.stan"
            )
        all_methods_default_data = dict(
            mu_0=prior_mean,
            v_0=prior_var,
            a_0=reg_ig_prior_shape,
            b_0=reg_ig_prior_scale,
            N_h=hidden_size,
        )

    methods = {}
    if "Chaudhuri" in method_names:
        if task_name == "logReg" or task_name == "linReg":

            def public_fn(theta_dict):
                return theta_dict

        else:

            def public_fn(theta_dict):
                return {k: np.mean(v, axis=0) for k, v in theta_dict.items()}

        methods["Chaudhuri"] = {
            "default_data": {"w": 1, **all_methods_default_data},
            "stan_fn": wKLBayes_stan.optimize
            if task_name != "mlpReg"
            else wKLBayes_stan.sample,
            "stan_args": dict()
            if task_name != "mlpReg"
            else dict(
                chains=1,
                iter_warmup=n_warmup,
                iter_sampling=n_mcmc,
                show_progress=False,
            ),
            "public_fn": public_fn,
            "privatise_fn": lambda theta, epsilon: {
                k: v + np.random.laplace(0, 2 / (epsilon / prior_var), v.shape)
                for k, v in theta.items()
            },
        }
    if "BetaBayes" in method_names:

        def privatise_fn(theta_dict, _epsilon):
            idx = np.random.choice(n_mcmc, (), replace=True)
            return {k: v[idx] for k, v in theta_dict.items()}

        def public_fn(theta_dict, task_name):
            if task_name == "mlpReg":
                return theta_dict
            else:
                return {k: np.mean(v, axis=0) for k, v in theta_dict.items()}

        def beta_given_eps(eps, _d):
            if task_name == "logReg":
                return 2 * beta_w / eps

            elif task_name == "linReg" or task_name == "mlpReg":

                def optim_fn(beta_p):
                    return (
                        (
                            2
                            * beta_w
                            / (
                                beta_p
                                * (2 * np.pi) ** (beta_p / 2)
                                * prior_var ** (beta_p / 2)
                            )
                        )
                        - eps
                    ) ** 2

                mini = minimize(optim_fn, 0.5, method="BFGS", tol=1e-6)
                return mini.x.item()

        methods["BetaBayes"] = {
            "default_data": {
                "sigma2_lower": 1e-2,
                **all_methods_default_data,
            },
            "add_eps_data_fn": {
                "beta_p": beta_given_eps,
            },
            "stan_fn": betaBayes_stan.sample,
            "stan_args": dict(
                chains=1,
                iter_warmup=n_warmup,
                iter_sampling=n_mcmc,
                show_progress=True if task_name == "mlpReg" else False,
            ),
            "public_fn": functools.partial(public_fn, task_name=task_name),
            "privatise_fn": privatise_fn,
        }
    if "Minami" in method_names:

        def privatise_fn(theta_dict, _epsilon):
            idx = np.random.choice(n_mcmc, (), replace=True)
            return {k: v[idx] for k, v in theta_dict.items()}

        def beta_given_eps(eps, d):
            return (
                eps
                / (2 * np.sqrt(d))
                / np.sqrt((1 + 2 * np.log(1 / delta)))
                / np.sqrt(prior_var)
            )

        methods["Minami"] = {
            "default_data": all_methods_default_data,
            "add_eps_data_fn": {"w": beta_given_eps},
            "stan_fn": wKLBayes_stan.sample,
            "stan_args": dict(
                chains=1,
                iter_warmup=n_warmup,
                iter_sampling=n_mcmc,
                show_progress=False,
            ),
            "public_fn": lambda theta_dict: {
                k: np.mean(v, axis=0) for k, v in theta_dict.items()
            },
            "privatise_fn": privatise_fn,
        }
    if "Chaudhuri_sklearn" in method_names:

        def privatise_fn(theta_dict, epsilon, reg, n, d):
            priv_theta_dict = theta_dict.copy()

            # privatise
            for k, v in priv_theta_dict.items():
                lapl_scale = 2 / (np.sqrt(n) * reg * epsilon)
                lapl_noice = np.random.laplace(
                    loc=0, scale=lapl_scale, size=v.shape
                )
                if k != "sigma2":
                    priv_theta_dict[k] += lapl_noice
            return priv_theta_dict

        methods["Chaudhuri_sklearn"] = {
            "default_data": {},
            "add_eps_data_fn": {},
            "stan_fn": sklearn_reg,
            "stan_args": dict(reg=reg, task_name=task_name),
            "public_fn": lambda theta_dict: theta_dict,
            "privatise_fn": privatise_fn,
        }
    if "dpsgd" in method_names:
        from dpsgd_net import dpsgd_net

        methods["dpsgd"] = dict(
            stan_fn=dpsgd_net,
            default_data={},
            add_eps_data_fn={
                "eps": lambda eps, _d: eps,
            },
            stan_args=dict(
                lr=lr,
                clip_norm=clip_norm,
                epochs=epochs,
                batch_size=batch_size,
                delta=delta,
                hidden_size=hidden_size,
                run_bnn=run_bnn,
            ),
        )

    return methods


def create_loglik_fn(task_name, method_name, epsilon, beta_w=1):
    if task_name == "logReg":
        if method_name in ["Chaudhuri", "Chaudhuri_sklearn", "Minami"]:

            def loglik_fn(data, params):
                X, y = data.x, data.y
                lin_pred = X @ params["theta"]
                p = np.clip(1 + np.exp(-1 * y * lin_pred), 1e-6, None)
                return -np.log(p).mean() * beta_w

            # def loglik_fn(data, params):
            #     X, y = data.x, data.y
            #     lin_pred = X @ params["theta"]
            #     p = y * lin_pred
            #     return p.mean()

        elif method_name == "BetaBayes":

            def loglik_fn(data, params):
                X, y = data.x, data.y
                beta_p = 2 * beta_w / epsilon
                lin_pred = X @ params["theta"]
                p_logistic = np.exp(0.5 * y * lin_pred) / (
                    np.exp(0.5 * lin_pred) + np.exp(-0.5 * lin_pred)
                )

                target = 1 / beta_p * p_logistic**beta_p - 1 / (
                    beta_p + 1
                ) * (
                    p_logistic ** (beta_p + 1)
                    + (1 - p_logistic) ** (beta_p + 1)
                )
                return target.mean()

    elif task_name in ["linReg", "mlpReg"]:
        predict_fn = eval.create_predict_fn(task_name, method_name)

        if method_name in ["Chaudhuri", "Chaudhuri_sklearn", "Minami", "dpsgd"]:

            def loglik_fn(data, params):
                X, y = data.x, data.y
                pred = predict_fn(X, params)
                return norm(pred, np.sqrt(params["sigma2"])).logpdf(y).mean()

        elif method_name == "BetaBayes":

            def loglik_fn(data, params):
                X, y = data.x, data.y
                pred = predict_fn(X, params)
                beta_p = 2 * beta_w / epsilon
                target = (1 / beta_p) * np.exp(
                    norm(pred, np.sqrt(params["sigma2"])).logpdf(y)
                ) ** beta_p  # minus constant
                return target.mean()

    return loglik_fn


def fit(method_name, method_dict, data_X, data_y, epsilon_vec, seed, cache_dir):
    n_feats = data_X.shape[1]
    stan_default_data = dict(n=data_X.shape[0], p=n_feats, y=data_y, X=data_X)

    theta_hat = {}
    theta_hat_priv = {}

    if method_name == "Chaudhuri":
        # fitting is independent of epsilon
        method_fit = method_dict["stan_fn"](
            data={**stan_default_data, **method_dict["default_data"]},
            seed=seed,
            **method_dict["stan_args"],
            output_dir=cache_dir,
        )
        if type(method_fit) is collections.OrderedDict:
            params = method_fit
            params["theta"] = method_fit["theta"].reshape((-1,))
        else:
            params = method_fit.stan_variables()
        if "theta" in params and type(params["theta"]) == float:
            params["theta"] = np.array([params["theta"]])
        for e in epsilon_vec:
            theta_hat[f"{method_name}-{e}"] = method_dict["public_fn"](
                {k: np.array(v).copy() for k, v in params.items()}
            )
            theta_hat_priv[f"{method_name}-{e}"] = method_dict["privatise_fn"](
                {k: np.array(v).copy() for k, v in params.items()}, e
            )
            # if params is not theta, Chaudhuri likely doesnt apply

    elif method_name == "Chaudhuri_sklearn":
        # fitting is independent of epsilon
        params = method_dict["stan_fn"](
            data_X, data_y, **method_dict["stan_args"]
        )

        for e in epsilon_vec:
            theta_hat[f"{method_name}-{e}"] = method_dict["public_fn"](
                {k: np.array(v).copy() for k, v in params.items()}
            )
            theta_hat_priv[f"{method_name}-{e}"] = method_dict["privatise_fn"](
                {k: np.array(v).copy() for k, v in params.items()},
                e,
                reg=method_dict["stan_args"]["reg"],
                n=data_X.shape[0],
                d=data_X.shape[1],
            )
            # if params is not theta, Chaudhuri likely doesnt apply

    else:
        for e in epsilon_vec:
            method_fit = method_dict["stan_fn"](
                data={
                    **stan_default_data,
                    **method_dict["default_data"],
                    **{
                        k: v(e, n_feats)
                        for k, v in method_dict["add_eps_data_fn"].items()
                    },
                },
                seed=seed,
                **method_dict["stan_args"],
                output_dir=cache_dir,
            )
            if method_name != "dpsgd":
                params = method_fit.stan_variables()
                theta_hat[f"{method_name}-{e}"] = method_dict["public_fn"](
                    {k: np.array(v).copy() for k, v in params.items()}
                )
                theta_hat_priv[f"{method_name}-{e}"] = method_dict[
                    "privatise_fn"
                ]({k: np.array(v).copy() for k, v in params.items()}, e)
            else:
                params = method_fit
                theta_hat[f"{method_name}-{e}"] = params["nonpriv"]
                theta_hat_priv[f"{method_name}-{e}"] = params["priv"]

    ori_samples = copy.copy(params)

    return theta_hat, theta_hat_priv, ori_samples


def sklearn_reg(
    X,
    y,
    reg,
    task_name,
):
    if task_name == "logReg":
        discr = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1 / (reg * np.sqrt(len(X))),
            # max_iter=4000,
            fit_intercept=False,
        )
    elif task_name == "linReg":
        discr = Ridge(
            alpha=reg,
            fit_intercept=False,
        )
    elif task_name == "mlpReg":
        discr = MLPRegressor(
            hidden_layer_sizes=(10,),
            activation="relu",
            solver="sgd",
            batch_size=X.shape[1],
            max_iter=10,
            learning_rate_init=1e-3,
            alpha=0,
            momentum=0.9,
            # best args
            # solver="lbfgs",
            # max_iter=4000,
        )
        discr.fit(X, y)
        theta = {
            "w_input": discr.coefs_[0],
            "b_input": discr.intercepts_[0],
            "w_output": discr.coefs_[1].reshape((-1,)),
            "b_output": discr.intercepts_[1].reshape((-1,)),
        }
        return theta

    discr.fit(X, y)

    return {"theta": discr.coef_.reshape((-1,)), "sigma2": 1}
