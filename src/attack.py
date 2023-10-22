import numpy as np
import scipy.stats
import math
from tqdm import tqdm
import warnings

import utils


def clopper_pearson(x, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


def model_train(train_fn, d0, d1, idx_round):
    idx = np.random.randint(2)
    d = [d0, d1][idx]
    trained = train_fn(d, idx_round)
    return idx, trained


def distinguisher(trained, d0, d1, score_fn, train_fn, norm_const):
    l0 = score_fn(d0, trained)
    l1 = score_fn(d1, trained)
    return (np.exp(l0 - l1) * norm_const) < 1

    # following nearly never occurs
    if (l0 != l0 and l1 != l1) or l0 == l1:
        print("Warning: equal scores")
        d0_trained = [
            np.concatenate([t.reshape(-1) for t in train_fn(d0, i).values()])
            for i in range(100)
        ]
        d1_trained = [
            np.concatenate([t.reshape(-1) for t in train_fn(d1, i).values()])
            for i in range(100)
        ]
        theta = np.concatenate([t.reshape(-1) for t in trained.values()])
        score0 = -((d0_trained - theta) ** 2).mean()
        score1 = -((d1_trained - theta) ** 2).mean()
        if score0 > score1:
            return 0
        elif score0 < score1:
            return 1
        else:
            print("Warning: DOUBLE equal scores")
            return np.random.randint(2)

    else:
        return np.argmax([l0, l1])


def lower_bound_audit_nasr(
    train_fn,
    score_fn,
    craft_data_fn,
    n_rounds,
    delta,
    eval_fn=None,
    norm_const=1,
):
    # never possible to get eps higher than 5.6 with 1000 trials

    attack_data = np.zeros((n_rounds, 2))
    eval_metric = np.zeros((n_rounds))

    with warnings.catch_warnings():
        utils.blockPrint()
        for idx_round in tqdm(range(n_rounds)):
            j = 0
            d0, d1, eval_ds = craft_data_fn(idx_round)
            while len(d0.y) > 1 and (
                (d0.y[0] == d0.y).all() or (d1.y[0] == d1.y).all()
            ):
                j += 1
                d0, d1, eval_ds = craft_data_fn(idx_round + n_rounds + j)
            attack_data[idx_round, 0], trained = model_train(
                train_fn, d0, d1, idx_round
            )
            attack_data[idx_round, 1] = distinguisher(
                trained, d0, d1, score_fn, train_fn, norm_const
            )
            if eval_fn is not None:
                eval_metric[idx_round] = eval_fn(trained, eval_ds)
            # utils.distringuish_reg_plot(
            #     d0, d1, int(attack_data[idx_round, 0]), trained["theta"]
            # )
            # attack_data[idx_round, 1] = np.random.uniform() > 0.5
        utils.enablePrint()

    fp = np.sum(
        (attack_data[:, 0] != attack_data[:, 1]) * (attack_data[:, 1] == 1)
    )
    fn = np.sum(
        (attack_data[:, 0] != attack_data[:, 1]) * (attack_data[:, 1] == 0)
    )

    # copper pearson
    _, fp_high = clopper_pearson(fp, n_rounds)
    _, fn_high = clopper_pearson(fn, n_rounds)

    lower_eps = max(
        np.log((1 - delta - fp_high) / fn_high),
        np.log((1 - delta - fn_high) / fp_high),
    )

    # n_rounds = 500
    # delta = 0
    # max_high = hi_clopper_pearson(0, n_rounds, 0.05)
    # max_lower_eps = max(
    #     np.log((1 - delta - max_high) / max_high),
    #     np.log((1 - delta - max_high) / max_high),
    # )
    # print(max_lower_eps)

    return (
        lower_eps,
        fp / n_rounds,
        fn / n_rounds,
        (attack_data[:, 0] == attack_data[:, 1]).mean(),
        eval_metric.mean(),
        eval_metric.std(),
    )
