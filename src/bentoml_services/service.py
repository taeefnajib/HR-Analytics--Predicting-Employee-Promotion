"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


model_runner = bentoml.sklearn.get("hr_model:latest").to_runner()
svc = bentoml.Service("hr_model", runners=[model_runner]) # it's important that this is assinged to a variable named "svc"


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def hr_model(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result