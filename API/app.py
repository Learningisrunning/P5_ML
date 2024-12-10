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

    input_data = {
                    "Title": "Je suis un titre test",
                    "Body": "Python, youtube et youtube et youutbe et zoom"
    }
    
    input_data = pd.DataFrame([input.dict()])

    print("Données d'entrée:", input_data)

    try:
        catalog = session.load_context().catalog
        catalog.save("input_data", input_data) 

        # Exécuter le pipeline "predict_pipeline"
        result = session.run(pipeline_name="predict_pipeline")

        # Récupérer les prédictions
        predictions = result.get("prediction") 
        if predictions is not None:
            return {"predictions": predictions}
        else:
            raise HTTPException(status_code=500, detail="No predictions returned by the pipeline.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
