import joblib
import bentoml
with open("/userRepoData/taeefnajib/HR-Analytics--Predicting-Employee-Promotion/__sidetrek__/models/828a95ed46728d5ab51e638d0d391c9c.joblib", 'rb') as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "hr_model2",
        model,
    )
    print(saved_model) # This is required!
