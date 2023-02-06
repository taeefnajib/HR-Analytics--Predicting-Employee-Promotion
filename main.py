# Importing all dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    data_filepath: str = "data/raw.csv"
    model_filepath: str = "model/model.pkl"
    test_size: float = 0.25
    random_state: int = 6
    learning_rate: float = 0.13
    n_estimators: int = 200
    min_samples_split: int = 3
    min_samples_leaf: int = 4
    max_depth: int = 5

hp = Hyperparameters()

# Creating the dataframe
def create_df(data_filepath):
    return pd.read_csv(data_filepath)

# Cleaning dataset
def clean_ds(df):
    df.drop([45868], inplace=True)
    df.drop(["employee_id"], axis=1, inplace=True)
    df["previous_year_rating"].fillna(0, inplace=True)
    df["education"].fillna("Unknown", inplace=True)
    return df

# Handling categorical columns
def handle_cat_cols(df):
    cat_cols = [x for x in df.columns if df[x].dtypes=="O"]
    df = pd.get_dummies(data=df, columns=cat_cols)
    return df

# Pre-processing dataset
def preprocess_ds(df):
    df = clean_ds(df)
    df = handle_cat_cols(df)
    df.to_csv("data/processed.csv", index=False)
    return df

# Splitting dataset into train, validation and test data
def split_train_test(df, test_size, random_state):
    X = df.drop(["is_promoted"], axis=1)
    y = df["is_promoted"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Training model on train data
def train_model(X_train, y_train,
                learning_rate, n_estimators,
                min_samples_split, min_samples_leaf,
                max_depth, random_state, model_filepath):
    gdc = GradientBoostingClassifier(
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf,
        max_depth = max_depth, 
        random_state=random_state,
    )
    #0.5408
    gdc.fit(X_train, y_train)
    pickle.dump(gdc, open(model_filepath, "wb"))
    return gdc

# Predicting on test data
def predict(X_test, y_test, model_filepath):
    model = pickle.load(open(model_filepath, "rb"))
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="binary")
    print(f"F1 Score for GradientBoostingClassifier: {f1}")
    return f1


def run_wf(hp: Hyperparameters) -> GradientBoostingClassifier:
    df = create_df(hp.data_filepath)
    df = preprocess_ds(df=df)
    X_train, X_test, y_train, y_test = split_train_test(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train,
                        learning_rate=hp.learning_rate,
                        n_estimators=hp.n_estimators,
                        min_samples_leaf=hp.min_samples_leaf,
                        min_samples_split=hp.min_samples_split,
                        max_depth=hp.max_depth,
                        random_state=hp.random_state,
                        model_filepath=hp.model_filepath)

# f1 = predict(X_test=X_test, y_test=y_test, model_filepath=hp.model_filepath)
    

if __name__=="__main__":
    run_wf(hp=hp)