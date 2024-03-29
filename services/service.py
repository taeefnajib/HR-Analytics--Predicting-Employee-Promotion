# THIS IS JUST AN EXAMPLE!
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray



model_runner = bentoml.sklearn.get("hr_model3:latest").to_runner()
svc = bentoml.Service("hr_model3", runners=[model_runner])


def preprocess_test_data(test_data: np.ndarray) -> np.ndarray:

    bento_model: bentoml.Model = bentoml.models.get("hr_model3:latest")
    encoder = bento_model.custom_objects.get("encoder")
    cols = bento_model.custom_objects.get("cols")
    X_cols = bento_model.custom_objects.get("X_cols")

    df = pd.DataFrame(test_data, columns=cols[:-1])
    df.drop(["employee_id"], axis=1, inplace=True)
    cat_cols = [x for x in df.columns if df[x].dtypes=="O"]
    cat_df = df[cat_cols]
    encoded_data = encoder.fit_transform(cat_df)
    one_hot_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    df = df.drop(cat_cols, axis=1)
    df = pd.concat([df, one_hot_df], axis=1)
    df1 = pd.DataFrame(0, index=[0], columns=X_cols)
    df1.update(df)
    return df1.values

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    input_series_processed = preprocess_test_data(test_data=input_series)
    result = model_runner.predict.run(input_series_processed)
    return result