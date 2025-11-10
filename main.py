from fastapi import FastAPI
import joblib
model = joblib.load("mymodel")

app = FastAPI()

@app.get("/")
def predict_iris(sl:float,sw:float,pl:float,pw:float):
    result = model.predict([[sl,sw,pl,pw]])
    return{"prediction is ":int(result[0])}