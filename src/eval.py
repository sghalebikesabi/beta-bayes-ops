import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def rmse_loss(preds, targets, axis=None):
    assert preds.shape == targets.shape
    rmse = np.sqrt(
        np.sum(np.square(preds - targets), axis=axis) / preds.shape[0]
    )
    if len(preds.shape) == 2:
        return rmse / preds.shape[1]
    else:
        return rmse


def mae_loss(preds, targets, axis=None):
    assert preds.shape == targets.shape
    return np.sum(np.abs(preds - targets), axis=axis) / np.prod(preds.shape)


def correct_sign_accuracy(theta_hat, theta_gen, _axis=None):
    assert theta_hat.shape == theta_gen.shape
    return np.mean((theta_hat > 0) == (theta_gen > 0))


def balanced_accuracy(preds, targets):
    assert preds.shape == targets.shape
    return balanced_accuracy_score(
        targets, (preds > 0.5) * 2 - 1, sample_weight=None
    )


def roc_auc(preds, targets):
    assert preds.shape == targets.shape
    return roc_auc_score(targets, preds, average="macro", sample_weight=None)


def logisitc_to_prob(Xtheta):
    # Function to calculate logistic regression probability
    Xtheta = np.clip(Xtheta, None, 1e3)
    return np.exp(0.5 * Xtheta) / (np.exp(0.5 * Xtheta) + np.exp(-0.5 * Xtheta))


def logReg_predict(X, theta):
    # Function to calculate logistic regression probability
    Xtheta = X @ theta["theta"]
    return logisitc_to_prob(Xtheta)


def linReg_predict(X, theta):
    return X @ theta["theta"]


def mlpReg_predict(X, theta):
    if len(theta["w_input"].shape) == 2:
        hidden = np.maximum(0, X @ theta["w_input"] + theta["b_input"])
        return hidden @ theta["w_output"] + theta["b_output"]
    else:
        hidden = np.maximum(0, X @ theta["w_input"] + theta["b_input"][:, None])
        return (
            np.einsum("ijk,ik->ij", hidden, theta["w_output"], optimize=True)
            + theta["b_output"][:, None]
        ).mean(0)


def create_predict_fn(task_name, method_name):
    predict_dict = {
        "logReg": logReg_predict,
        "linReg": linReg_predict,
        "mlpReg": mlpReg_predict,
    }
    if task_name == "mlpReg" and method_name == "dpsgd":
        import dpsgd_net

        return dpsgd_net.predict
    else:
        return predict_dict[task_name]


def eval(
    task_name, method_name, theta_hat, theta_hat_priv, eval_dataset, suffix=""
):
    predict_fn = create_predict_fn(task_name, method_name)
    if task_name in ["linReg", "mlpReg"]:
        metrics_fns = {"mae": mae_loss, "rmse": rmse_loss}
    else:
        metrics_fns = {"acc": balanced_accuracy, "auc": roc_auc}

    metrics = {}
    metrics_priv = {}

    for metrics_name, fn in metrics_fns.items():
        assert len(eval_dataset.y_eval.shape) < 3
        pred = predict_fn(eval_dataset.x_eval, theta_hat)
        pred_priv = predict_fn(eval_dataset.x_eval, theta_hat_priv)

        metrics[f"{metrics_name}{suffix}"] = fn(pred, eval_dataset.y_eval)
        metrics_priv[f"{metrics_name}{suffix}"] = fn(
            pred_priv, eval_dataset.y_eval
        )

    if (
        getattr(eval_dataset, "theta_gen", None) is not None
        and task_name != "mlpReg"
    ):
        theta_hat_keys = (
            [  # relevant theta keys that we want to estimate correctly
                t
                for t in theta_hat.keys()
                if t not in ["sigma2", "lp__", "lin_pred", "int_term"]
            ]
        )
        metrics_fns = {
            "mae": mae_loss,
            "rmse": rmse_loss,
            "csa": correct_sign_accuracy,
        }
        for metrics_name, fn in metrics_fns.items():
            for theta_name in theta_hat_keys:  # different parameters
                assert len(theta_hat[theta_name].shape) < 3
                metrics[f"{metrics_name}_{theta_name}{suffix}"] = fn(
                    theta_hat[theta_name],
                    eval_dataset.theta_gen[theta_name],
                    None,
                )
                metrics_priv[f"{metrics_name}_{theta_name}{suffix}"] = fn(
                    theta_hat_priv[theta_name],
                    eval_dataset.theta_gen[theta_name],
                    None,
                )

    return metrics, metrics_priv


def create_eval_df(
    metrics,
    metrics_priv,
):
    loss_lst = []
    for loss_dict, loss_suffix in zip(
        [metrics, metrics_priv],
        ["", "_priv"],
    ):
        for loss_name, loss in loss_dict.items():
            metric, k, j, d, n, e = loss_name.split("-")
            loss_lst.append(
                {
                    "n": n,
                    "epsilon": e,
                    "loss": metric + loss_suffix,
                    "rep": j,
                    "method": k,
                    "value": loss,
                    "n_feats": d,
                }
            )
    loss_df = pd.DataFrame.from_records(loss_lst)
    return loss_df
