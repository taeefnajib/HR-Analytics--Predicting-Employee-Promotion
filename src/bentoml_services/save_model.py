"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import joblib
import bentoml


with open("/userRepoData/__sidetrek__/taeefnajib/HR-Analytics--Predicting-Employee-Promotion/bentoml/models/666e50772909e321da2bafdeab225fda.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "hr_model",
        model,
    )
    print(saved_model) # This is required!