# Importing all dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from sidetrek.types.dataset import SidetrekDataset
from sidetrek.dataset import load_dataset
from typing import Tuple

@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.25
    random_state: int = 6
    learning_rate: float = 0.13
    n_estimators: int = 200
    min_samples_split: int = 3
    min_samples_leaf: int = 4
    max_depth: int = 5

hp = Hyperparameters()

# Creating the dataframe
def create_df(ds: SidetrekDataset) -> pd.DataFrame:
    data = load_dataset(ds = ds, data_type="csv")
    data_dict = {}
    cols = list(data)[0]
    for k, v in enumerate(data):
        if k>0:
            data_dict[k]=v
    df = pd.DataFrame.from_dict(data_dict, columns=cols, orient="index")
    return df

# Cleaning dataset
def clean_ds(df: pd.DataFrame) -> pd.DataFrame:
    df.drop([45868], inplace=True)
    df.drop(["employee_id"], axis=1, inplace=True)
    df["previous_year_rating"].fillna(0, inplace=True)
    df["education"].fillna("Unknown", inplace=True)
    df["is_promoted"]=df["is_promoted"].astype(int)
    return df

# Handling categorical columns
def handle_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [x for x in df.columns if df[x].dtypes=="O"]
    df = pd.get_dummies(data=df, columns=cat_cols)
    return df

# Splitting dataset into train, validation and test data
def split_train_test(df: pd.DataFrame, hp: Hyperparameters) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(["is_promoted"], axis=1)
    y = df["is_promoted"]
    return train_test_split(X, y, test_size=hp.test_size, random_state=hp.random_state)


# Training model on train data
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    gdc = GradientBoostingClassifier(
        learning_rate = hp.learning_rate,
        n_estimators = hp.n_estimators,
        min_samples_split = hp.min_samples_split,
        min_samples_leaf = hp.min_samples_leaf,
        max_depth = hp.max_depth, 
        random_state=hp.random_state,
    )
    #0.5408
    gdc.fit(X_train, y_train)
    return gdc

# # Predicting on test data
# def predict(X_test, y_test, model_filepath):
#     model = pickle.load(open(model_filepath, "rb"))
#     y_pred = model.predict(X_test)
#     f1 = f1_score(y_test, y_pred, average="binary")
#     print(f"F1 Score for GradientBoostingClassifier: {f1}")
#     return f1


# def run_wf(hp: Hyperparameters) -> GradientBoostingClassifier:
#     df = create_df(hp.data_filepath)
#     df = preprocess_ds(df=df)
#     X_train, X_test, y_train, y_test = split_train_test(df=df, test_size=hp.test_size, random_state=hp.random_state)
#     return train_model(X_train=X_train, y_train=y_train,
#                         learning_rate=hp.learning_rate,
#                         n_estimators=hp.n_estimators,
#                         min_samples_leaf=hp.min_samples_leaf,
#                         min_samples_split=hp.min_samples_split,
#                         max_depth=hp.max_depth,
#                         random_state=hp.random_state,
#                         model_filepath=hp.model_filepath)

# # f1 = predict(X_test=X_test, y_test=y_test, model_filepath=hp.model_filepath)
    

# if __name__=="__main__":
#     run_wf(hp=hp)