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
    vn: float
    zcr: float
    sf: float
    cgs: float
    snr: float
    cs: float


@app.post("/predict/")
def predict(input_data: InputData):
    # Convertir les données en tableau numpy
    input_array = np.array([[input_data.vn, input_data.zcr, input_data.sf,
                             input_data.cgs, input_data.snr, input_data.cs]])

    class_pred = classification_model.predict(input_array)
    classes = {0: "environnement", 1: "grésillement", 2: "souffle"}
    reg_pred = regression_model.predict(scaler.transform(input_array))
    return {"quality": reg_pred[0], "class": classes[int(class_pred[0])]}
