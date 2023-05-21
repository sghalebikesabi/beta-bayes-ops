"""UCI Datasets.

This file contains the code to load the UCI datasets. Mostly adapted from https://github.com/MrHuff/GWI/blob/7321cc7a3ae6566d9c1f145a65d7707a00d6384f/utils/regression_dataloaders.py.
"""
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import zipfile
from .wrapper import AbstractDataset


import utils


class UCIDataset(AbstractDataset):
    """UCI Datasets.

    # TODO: Each dataset is split into 20 train-test folds, except for the protein dataset which uses 5 folds and the Year Prediction MSD dataset which uses a single train-test split.

    """

    def __init__(
        self,
        task_name,
        name,
        seed,
        eval_split,
        add_constant,
        n_feats,
        n_samples=None,
        n_samples_eval=None,
        hidden_size=None,
    ):
        del n_samples
        del n_samples_eval
        del hidden_size

        self.name = name
        self.seed = seed

        self.datasets = {
            # classification
            "abalone": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
            "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "balloon": "https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data",  # 20 observations
            "bank": "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
            "breast": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
            "car": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",  # three classes
            "chess": "https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data",
            "statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",  # too many attributes
            "raisin": "https://archive.ics.uci.edu/ml/machine-learning-databases/00617/Raisin_Dataset.zip",
            "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            "vertebral": "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip",
            # regression
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            "boston": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "naval": "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
            "KIN8NM": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
            "protein": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        }
        self.data_path = utils.get_data_root() + "/"
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        data = self._load_dataset(task_name)
        data = data[:, np.std(data, 0) > 0]

        self.out_dim = 1
        self.n_feats = data.shape[1] - self.out_dim

        x, x_eval, self.y, self.y_eval = self._get_splits(
            data, eval_split, task_name
        )
        scaler = MinMaxScaler()
        self.x = scaler.fit_transform(x)
        self.x_eval = scaler.transform(x_eval)

        self.n_samples = len(self.x)

        if n_feats != "None":
            feat_index = np.random.choice(
                self.x.shape[1], n_feats - add_constant, replace=False
            )
            self.x = self.x[:, feat_index]
            self.x_eval = self.x_eval[:, feat_index]
            self.n_feats = self.x.shape[1]

        if add_constant:
            self.x = np.hstack((np.ones((self.n_samples, 1)), self.x))
            self.x_eval = np.hstack(
                (np.ones((self.x_eval.shape[0], 1)), self.x_eval)
            )

    def _load_dataset(self, task_name):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not os.path.exists(self.data_path + "UCI"):
            os.mkdir(self.data_path + "UCI")

        url = self.datasets[self.name]
        file_name = url.split("/")[-1]

        if not os.path.exists(self.data_path + "UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path + "UCI/" + file_name
            )

        # -------------------------CLASSIFICATION DATASETS-------------------------#
        if self.name == "abalone":
            assert task_name == "logReg"
            data = pd.read_csv(
                self.data_path + "UCI/abalone.data",
                header=None,  # delimiter="\s+"
            )
            data.columns = [
                "sex",
                "length",
                "diameter",
                "height",
                "whole_weight",
                "shucked_weight",
                "viscera_weight",
                "shell_weight",
                "rings",
            ]
            data = pd.concat(
                [pd.get_dummies(data["sex"], prefix="sex"), data], axis=1
            )
            data.drop(["sex"], axis=1, inplace=True)
            # target variable encoding
            target_col = data.columns[-1]
            ring_threshold = 10
            data[target_col] = np.where(data[target_col] > ring_threshold, 1, 0)

            data = data.values[np.random.permutation(np.arange(len(data)))]

        elif self.name == "adult":
            assert task_name == "logReg"
            data_types = OrderedDict(
                [
                    ("age", "int"),
                    ("workclass", "category"),
                    ("final_weight", "int"),  # originally it was called fnlwgt
                    ("education", "category"),
                    ("education_num", "int"),
                    ("marital_status", "category"),
                    ("occupation", "category"),
                    ("relationship", "category"),
                    ("race", "category"),
                    ("sex", "category"),
                    ("capital_gain", "float"),  # required because of NaN values
                    ("capital_loss", "int"),
                    ("hours_per_week", "int"),
                    ("native_country", "category"),
                    ("income_class", "category"),
                ]
            )
            target_column = "income_class"

            def read_dataset(path):
                return pd.read_csv(
                    path,
                    names=data_types,
                    index_col=None,
                    comment="|",  # test dataset has comment in it
                    skipinitialspace=True,  # Skip spaces after delimiter
                    na_values={
                        "capital_gain": 99999,
                        "workclass": "?",
                        "native_country": "?",
                        "occupation": "?",
                    },
                    dtype=data_types,
                )

            def clean_dataset(data):
                # Test dataset has dot at the end, we remove it in order
                # to unify names between training and test datasets.
                data["income_class"] = data.income_class.str.rstrip(".").astype(
                    "category"
                )

                # Remove final weight column since there is no use
                # for it during the classification.
                data = data.drop("final_weight", axis=1)

                # # Duplicates might create biases during the analysis and
                # # during prediction stage they might give over-optimistic
                # # (or pessimistic) results.
                # data = data.drop_duplicates()

                # Binarize target variable (>50K == 1 and <=50K == 0)
                data[target_column] = (data[target_column] == ">50K").astype(
                    int
                )

                return data

            def get_categorical_columns(data, cat_columns=None, fillna=True):
                if cat_columns is None:
                    cat_data = data.select_dtypes("category")
                else:
                    cat_data = data[cat_columns]

                if fillna:
                    for colname in cat_data:
                        series = cat_data[colname]
                        if "Other" not in series.cat.categories:
                            series = series.cat.add_categories(["Other"])

                        cat_data[colname] = series.fillna("Other")

                return cat_data

            def features_with_one_hot_encoded_categories(
                data, cat_columns=None, fillna=True
            ):
                cat_data = get_categorical_columns(data, cat_columns, fillna)
                one_hot_data = pd.get_dummies(cat_data)
                df = pd.concat([data, one_hot_data], axis=1)

                features = [
                    "age",
                    "education_num",
                    "hours_per_week",
                    "capital_gain",
                    "capital_loss",
                ]  # ! + one_hot_data.columns.tolist()

                X = df[features].fillna(0).values.astype(float)
                y = df[target_column].values

                return np.concatenate([X, y[:, None]], axis=1)

            data = read_dataset(self.data_path + "UCI/adult.data")
            data = clean_dataset(data)
            data = features_with_one_hot_encoded_categories(data)

            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "balloon":
            assert task_name in ["logReg"]
            data = pd.read_csv(
                self.data_path + "UCI/adult+stretch.data", header=None
            )
            data = pd.get_dummies(data, drop_first=True).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "bank":
            assert task_name in ["logReg"]
            data = pd.read_csv(
                self.data_path + "UCI/data_banknote_authentication.txt",
                header=None,
                delimiter=",",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "breast":
            assert task_name in ["logReg"]
            data = pd.read_csv(
                self.data_path + "UCI/breast-cancer.data", header=None
            )
            data = pd.get_dummies(data, drop_first=True).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "car":
            assert task_name in ["logReg"]
            data = pd.read_csv(self.data_path + "UCI/car.data", header=None)
            data = pd.get_dummies(data, drop_first=True).values
            data = data[np.random.permutation(np.arange(len(data)))]

        # elif self.name == "statlog":
        #     assert task_name in ["logReg"]
        #     data = pd.read_csv(
        #         self.data_path + "UCI/german.data-numeric",
        #         header=None,
        #         sep="\t",
        #     )
        #     data = pd.get_dummies(data, drop_first=True).values
        #     data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "chess":
            assert task_name in ["logReg"]
            data = pd.read_csv(
                self.data_path + "UCI/kr-vs-kp.data", header=None
            )
            data = pd.get_dummies(data, drop_first=True).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "vertebral":
            assert task_name in ["logReg"]
            zipfile.ZipFile(
                self.data_path + "UCI/vertebral_column_data.zip"
            ).extractall(self.data_path + "UCI/vertebral/")
            data = pd.read_csv(
                self.data_path + "UCI/vertebral/column_2C.dat",
                header=0,
                delimiter=" ",
            )
            data.iloc[:, -1] = data.iloc[:, -1].map({"AB": 1, "NO": 0})
            data = data.values[np.random.permutation(np.arange(len(data)))]

        elif self.name == "parkinsons":
            assert task_name in ["logReg"]
            data = pd.read_csv(
                self.data_path + "UCI/parkinsons.data", header=0, index_col=0
            ).values
            data[:, [16, -1]] = data[:, [-1, 16]]
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "raisin":
            zipfile.ZipFile(
                self.data_path + "UCI/Raisin_Dataset.zip"
            ).extractall(self.data_path + "UCI/raisin/")
            data = pd.read_excel(
                self.data_path
                + "UCI/raisin/Raisin_Dataset/Raisin_Dataset.xlsx",
                header=0,
                engine="openpyxl",
            )
            data.iloc[:, -1] = data.iloc[:, -1].map({"Kecimen": 0, "Besni": 1})
            data = data.values[np.random.permutation(np.arange(len(data)))]

        # -------------------------REGRESSION DATASETS-------------------------#
        elif self.name == "boston":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_csv(
                self.data_path + "UCI/housing.data",
                header=None,
                delimiter="\s+",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_excel(
                self.data_path + "UCI/Concrete_Data.xls",
                header=0,
                # engine="openpyxl",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "energy":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_excel(
                self.data_path + "UCI/ENB2012_data.xlsx",
                header=0,
                engine="openpyxl",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "power":
            assert task_name in ["linReg", "mlpReg"]
            zipfile.ZipFile(self.data_path + "UCI/CCPP.zip").extractall(
                self.data_path + "UCI/CCPP/"
            )
            data = pd.read_excel(
                self.data_path + "UCI/CCPP/CCPP/Folds5x2_pp.xlsx",
                header=0,
                engine="openpyxl",
            ).values
            np.random.shuffle(data)

        elif self.name == "protein":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_csv(
                self.data_path + "UCI/CASP.csv", header=0, delimiter=","
            ).iloc[:, ::-1]
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "wine":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_csv(
                self.data_path + "UCI/winequality-red.csv",
                header=0,
                delimiter=";",
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_csv(
                self.data_path + "UCI/yacht_hydrodynamics.data",
                header=None,
                delimiter="\s+",
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "naval":
            assert task_name in ["linReg", "mlpReg"]
            zipfile.ZipFile(
                self.data_path + "UCI/UCI%20CBM%20Dataset.zip"
            ).extractall(self.data_path + "UCI/UCI CBM Dataset/")
            data = pd.read_csv(
                self.data_path + "UCI/UCI CBM Dataset/UCI CBM Dataset/data.txt",
                header=None,
                delimiter="\s+",
            ).values
            data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "KIN8NM":
            assert task_name in ["linReg", "mlpReg"]
            data = pd.read_csv(
                self.data_path + "UCI/dataset_2175_kin8nm.csv", header=0
            )
            data = data.values
            data = data[np.random.permutation(np.arange(len(data)))]

        data = data.astype("float")
        if task_name == "logReg":
            data[:, -1] = data[:, -1] * 2 - 1

        return data

    def _get_splits(self, data, eval_split, task_name):
        train_valid_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=0.1,
            random_state=self.seed,
            shuffle=True,
        )
        x_train, y_train = (
            data[train_valid_idx, : self.n_feats],
            data[train_valid_idx, self.n_feats :],
        )
        # x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
        # x_stds[x_stds == 0] = 1.0
        # x_train = (x_train - x_means) / x_stds

        if eval_split == "test":
            x_eval = data[test_idx, : self.n_feats]
            y_eval = data[test_idx, self.n_feats :]

        elif eval_split == "valid":
            x_train, x_eval, y_train, y_eval = train_test_split(
                x_train, y_train, test_size=0.1
            )

        if task_name == "logReg":
            y_train, y_eval = y_train.astype("int"), y_eval.astype("int")
        y_train, y_eval = y_train.reshape(-1), y_eval.reshape(-1)

        return x_train, x_eval, y_train, y_eval
