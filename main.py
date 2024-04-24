from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="API de Clasificación de Especies de Flores Iris",
    version="1.0.0"
)

# Cargar el modelo entrenado
svm_model = joblib.load("model/svm_model.pkl")


@app.post("/api/v1/iris-classifier", tags=["iris-classifier"])
async def predict(
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
):
        directionary = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        try:
            df = pd.DataFrame(directionary, index=[0])
            prediction = svm_model.predict(df)
            return JSONResponse(
                status_code=200,
                content={"predicted_class": prediction}
            )
        except Exception as e:
            raise HTTPException(
                detail=str(e),
                status_code=status.HTTP_400_BAD_REQUEST
            )
