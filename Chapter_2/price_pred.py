from scipy import stats
import joblib
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import pandas as pd
from pathlib import Path
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import FunctionTransformer
data_path = Path("datasets/housing")


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    # Klasa ta przyjmuje współrzędne geograficzne i oblicza podobieństwo
    # do centroidów klastrów K-Means jako nowe cechy.
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        # Dopasowujemy K-Means
        self.kmeans_ = KMeans(
            self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        # Obliczamy podobieństwo RBF do każdego centroidu
        # To są nowe cechy dodane do modelu.
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    # Metoda wymagana do poprawnego działania ColumnTransformer w nowszych wersjach Scikit-learn
    def get_feature_names_out(self, input_features=None):
        return [f"cluster_sim_{i}" for i in range(self.n_clusters)]
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
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
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

# #? Simpler
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing['income_cat'], random_state=42)

# print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# * Visualizing geographical data

# housing.plot(kind="scatter", x="longitude", y='latitude', grid=True, s=housing['population']/100, label='population', c='median_house_value', cmap='jet', colorbar=True, legend=True, sharex=False, figsize=(10, 7))
# plt.show()

# corr_matrix = housing.corr(numeric_only=True)
# corr_matrix["median_house_value"].sort_values(ascending=False)
# attributes = ["median_house_value", "median_income", 'total_rooms', "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# housing.plot(kind="scatter", x="median_income",
#              y="median_house_value", alpha=0.1, grid=True)
# plt.show()

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedroms_ration"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_housee"] = housing["population"] / housing["households"]

# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy='median')
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(
    X, columns=housing_num.columns, index=housing_num.index)


housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head())

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# print(housing_cat_encoded[:8])
# print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# print(housing_cat_1hot.toarray())

# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# housing_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# std_scaler = StandardScaler()
# housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# target_scaler = StandardScaler()
# scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# model = LinearRegression()
# model.fit(housing_labels[["median_income"]], scaled_labels)
# some_new_data = housing[["median_income"]].iloc[:5]

# scaled_prediction = model.predict(some_new_data)
# predictions = target_scaler.inverse_transform(scaled_prediction)

# from sklearn.compose import TransformedTargetRegressor
# model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
# model.fit(housing[["median_income"]], housing_labels)

# some_new_data = housing[["median_income"]].iloc[:5]
# predictions = model.predict(some_new_data)

# from sklearn.preprocessing import FunctionTransformer
# log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
# log_pop = log_transformer.transform(housing[["population"]])

# num_pipeline = Pipeline([("impute", SimpleImputer(strategy='median')), ("standardize", StandardScaler())])

num_pipline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# housing_num_prepered = num_pipline.fit_transform(housing_num)
# print(housing_num_prepered[:2].round(2))
# df_housing_num_prepared = pd.DataFrame(housing_num_prepered, columns=num_pipline.get_feature_names_out(), index=housing_num.index)
# print(df_housing_num_prepared)

# num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
# cat_attribs = ["ocean_proximity"]
cat_pipeline = make_pipeline(SimpleImputer(
    strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
# preprocessing = ColumnTransformer([("num", num_pipline, cat_attribs), ('cat', cat_pipeline, cat_attribs)])

preprocessing = make_column_transformer(
    (num_pipline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

housing_prepared = preprocessing.fit_transform(housing)
# print(housing_prepared.shape)
# print(preprocessing.get_feature_names_out())
# print(pd.DataFrame(housing_prepared, columns=preprocessing.get_feature_names_out(), index=housing.index))


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                           "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline)
# * Train and Evaluate on the Training Set
# ? LinearRegression
# from sklearn.linear_model import LinearRegression

# lin_reg = make_pipeline(preprocessing, LinearRegression())
# lin_reg.fit(housing, housing_labels)

# housing_predictions = lin_reg.predict(housing)
# print(housing_predictions[:5].round(-2))
# print(housing_labels.iloc[:5].values)

# ? DecisionTreeRegression
# from sklearn.tree import DecisionTreeRegressor

# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# tree_reg.fit(housing, housing_labels)
# housing_predictions = tree_reg.predict(housing)

# ? RandomForestRegression
forest_reg = make_pipeline(
    preprocessing, RandomForestRegressor(random_state=42))

# rmse = root_mean_squared_error(housing_labels, housing_predictions)
# print(rmse)

# tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
# print(pd.Series(tree_rmses).describe())
# forest_rmse = -cross_val_score(forest_reg, housing, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
# print(pd.Series(forest_rmse).describe())

full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('random_forest', RandomForestRegressor(random_state=42))
])
parm_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]}
]
# grid_search = GridSearchCV(full_pipeline, parm_grid,
#                            cv=3, scoring='neg_root_mean_squared_error')
# grid_search.fit(housing, housing_labels)
# print(grid_search.best_params_)


param_distribs = {
    'preprocessing__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)
}
rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs,
                                n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)


final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
print(feature_importances.round(2))

sorted(zip(feature_importances,
       final_model["preprocessing"].get_feature_names_out()), reverse=True)

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
fianl_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, fianl_predictions)
print(final_rmse)


def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))


confidence = 0.95
squared_errors = (fianl_predictions - y_test) ** 2
boot_result = stats.bootstrap(
    [squared_errors], rmse, confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval

joblib.dump(final_model, "my_california_housing_model.pkl")
