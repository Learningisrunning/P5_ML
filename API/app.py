from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Initialiser le projet Kedro
PROJECT_PATH = "/app/kedro-pipeline"
bootstrap_project(PROJECT_PATH)


# Modèle pour les données d'entrée
class PredictionInput(BaseModel):
    Title: str
    Body: str

# Initialisation de FastAPI
app = FastAPI()

# Initialiser KedroSession au démarrage
session = KedroSession.create(project_path=PROJECT_PATH)

@app.post("/predict")
def predict(input: PredictionInput):
    # Convertir les données d'entrée en DataFrame
    input_data = pd.DataFrame([input.model_dump()])

    try:
        catalog = session.load_context().catalog
        catalog.save("input_data", input_data) 
        # Préparer les données d'entrée pour le pipeline
        #pipeline_inputs = {"input_data": input_data}  # Assurez-vous que "input_data" est le bon nom dans votre pipeline

        # Exécuter le pipeline "predict_pipeline"
        result = session.run(pipeline_name="predict_pipeline")

        # Récupérer les prédictions
        predictions = result.get("prediction")  # Remplacez "output_data" par le nom exact de votre sortie
        if predictions is not None:
            return {"predictions": predictions}
        else:
            raise HTTPException(status_code=500, detail="No predictions returned by the pipeline.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
