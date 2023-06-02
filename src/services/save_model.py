import joblib
import bentoml
from src.main import transforms


# with open("/userRepoData/taeefnajib/HR-Analytics--Predicting-Employee-Promotion/__sidetrek__/models/f395d4113c6342f2d2cc4f832eb8e18a.joblib", "rb") as f:
#     model = joblib.load(f)
#     saved_model = bentoml.sklearn.save_model(
#         "hr_model3",
#         model,
#         custom_objects= {
#             "encoder": transforms["encoder"],
#             # "cols" : cols,
#             # "X_cols": X_cols,
#         }
#     )
#     print(saved_model) # This is required!