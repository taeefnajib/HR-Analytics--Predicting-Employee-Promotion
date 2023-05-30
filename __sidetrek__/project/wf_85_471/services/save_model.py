import joblib
import bentoml
from project.wf_85_471.main import encoder, cols, X_cols


with open("/userRepoData/taeefnajib/HR-Analytics--Predicting-Employee-Promotion/__sidetrek__/models/828a95ed46728d5ab51e638d0d391c9c.joblib", 'rb') as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "hr_model2",
        model,
        custom_objects= {
        "encoder"  : encoder,
        "cols" : cols,
        "X_cols": X_cols,
        }
    )
    print(saved_model) # This is required!