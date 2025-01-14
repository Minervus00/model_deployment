import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Charger les modèles
with open('classif_modele.sav', 'rb') as f:
    classification_model = pickle.load(f)

with open('reg_bagging.sav', 'rb') as f:
    regression_model = pickle.load(f)

with open('reg_scaler.sav', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Schéma des données d'entrée
class InputData(BaseModel):
    # type: str  # "classification" ou "regression"
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float


@app.post("/predict/")
def predict(input_data: InputData):
    # Convertir les données en tableau numpy
    input_array = np.array([[input_data.feature1, input_data.feature2, input_data.feature3,
                             input_data.feature4, input_data.feature5, input_data.feature6]])

    class_pred = classification_model.predict(input_array)
    classes = {0: "environnement", 1: "grésillement", 2: "souffle"}
    reg_pred = regression_model.predict(scaler.transform(input_array))
    return {"classe": classes[int(class_pred[0])], "quality": reg_pred[0]}

# Vérifier le modèle à utiliser
# if input_data.type == "classification":
#     prediction = classification_model.predict(input_array)
#     return {"model": "classification", "prediction": int(prediction[0])}
# elif input_data.type == "regression":
#     prediction = regression_model.predict(input_array)
#     return {"model": "regression", "prediction": float(prediction[0])}
# else:
#     return {"error": "Type de modèle non valide. Utilisez 'classification' ou 'regression'."}
