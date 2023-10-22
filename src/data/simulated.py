import numpy as np

from .wrapper import AbstractDataset
import eval


def min_max_std_data(X, min_val, max_val):
    # Function to standardize data
    X = np.clip(X, min_val, max_val)
    X = (X - min_val) / (max_val - min_val)
    return X


def sample_normal_features(n_samples, n_feats, add_constant):
    if add_constant:
        n_feats = n_feats - 1

    predictor_covariance = np.diag(np.ones(n_feats))
    features = np.random.multivariate_normal(
        mean=np.zeros(n_feats), cov=predictor_covariance, size=n_samples
    )
    data_X = min_max_std_data(features, min_val=-4, max_val=4)

    if add_constant:
        data_X = np.hstack((np.ones((n_samples, 1)), data_X))

    return data_X


def sample_parameters(data_X, task_name, n_feats, hidden_size):
    if n_feats == 1:
        theta_gen = {"theta": np.array((9,))}
    elif task_name == "logReg" or task_name == "linReg":
        unstand_params = np.random.randn(n_feats) * 9
        unstand_params[0] -= (data_X @ unstand_params).mean()
        theta_gen = {"theta": unstand_params}
    elif task_name == "mlpReg":
        theta_gen = {
            "w_input": np.random.randn(n_feats, hidden_size),
            "b_input": np.random.randn(hidden_size),
            "w_output": np.random.randn(hidden_size),
            "b_output": np.random.randn(),
        }
        theta_gen["sigma2"] = np.exp(np.random.randn())
    else:
        raise NotImplementedError

    if task_name in ["linReg", "mlpReg"]:
        theta_gen["sigma2"] = np.exp(np.random.randn())

    return theta_gen


def sample_target(task_name, data_X, theta_gen):
    data_y = eval.create_predict_fn(task_name, "")(data_X, theta_gen)
    if task_name == "logReg":
        # data_y = (0.5 < data_y) * 2 - 1
        data_y = (np.random.uniform(size=(len(data_y),)) < data_y) * 2 - 1
    else:
        data_y = data_y + np.random.randn(*data_y.shape) * np.sqrt(
            theta_gen["sigma2"]
        )
    return data_y


class SimulatedDataset(AbstractDataset):
    def __init__(
        self,
        task_name,
        n_samples,
        n_samples_eval,
        add_constant,
        seed,
        name,
        eval_split,
        hidden_size,
        **data_kwargs
    ):
        del name
        del eval_split
        np.random.seed(seed)

        self.x = sample_normal_features(
            n_samples, add_constant=add_constant, **data_kwargs
        )
        self.theta_gen = sample_parameters(
            self.x, task_name, self.x.shape[1], hidden_size
        )
        self.y = sample_target(task_name, self.x, self.theta_gen)
        self.n_samples, self.n_feats = self.x.shape

        if n_samples_eval:
            self.x_eval = sample_normal_features(
                n_samples_eval, add_constant=add_constant, **data_kwargs
            )
            self.y_eval = sample_target(task_name, self.x_eval, self.theta_gen)
