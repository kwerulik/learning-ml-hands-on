import sklearn
import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data_path = Path("datasets/housing")

# * Loading data


def load_housing_data():
    tarball_path = Path("Chapter_2/datasets/housing.tgz")

    if not tarball_path.is_file():
        Path("Chapter_2/datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="Chapter_2/datasets")

    return pd.read_csv(Path("Chapter_2/datasets/housing/housing.csv"))


housing = load_housing_data()
# * Quick look at the data
#
# print(housing.head(10))
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(12, 8))
# plt.show()

# * Createing a test set

# ? without sklearn
# def shuffle_and_split_data(data, test_ratio):
#     shuffle_indices = np.random.premutatuin(len(data))
#     test_set_size = int(len(data * test_ratio))
#     test_indices = shuffle_indices[:test_set_size]
#     train_indices = shuffle_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = train_test_split(housing, test_size= 0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"], bins=[
                               0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# # plt.xlabel("Income category")
# # plt.ylabel("Number of districts")
# # plt.show()

# splitter = StratifiedShuffleSplit(n_splits =10, test_size=0.2, random_state=42)
# strat_splits = []

# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set_n, strat_test_set_n = strat_splits[0]

# #* Simpler
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing['income_cat'], random_state=42)

# print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
