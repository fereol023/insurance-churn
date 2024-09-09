import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from fastapi import FastAPI, HTTPException
from api.ClientDataModel import ClientDataM
from api.PredictController import ChurnC
import ml_pipelines
import subprocess

import shap
from dstoolbox.pipeline import DataFrameFeatureUnion

app = FastAPI(
    title="insurance churn-API",
    description="pour exposer le modèle de ML",
    version="POC",
)


@app.get("/", response_model=dict, tags=['Entrypoint'])
async def start():
    """TODO
    """
    return {
        "Message pour vous": "Bonjour",
        "Politique de confidentialité": "A voir",
        "Dernière màj": "06/03/2024",
        #'header': req.headers.get('authorization'), # enlever
        #"API Key": generateJWT(role_id="nightOnWebSiteApp"),
        "API Key": 'Not available.'
    }


@app.post("/call", tags=["modèle"])
async def call(data: ClientDataM):
    """
    Nouveau prédict utilisateur.
    --------------------------------------
    """
    c = ChurnC()
    c.predict(data)
    res = c.get_output()
    if res=="":
        raise HTTPException(status_code=500, detail="E.")
    return {"predict": res}

@app.get("/test", tags=["backoffice"])
async def call():
    """
    """
    return ChurnC().debug()

@app.get("/retrain_model", tags=["backoffice"])
async def retrain_model():
    """
    rerun train model from api 
    """
    ml_pipelines.run_script('train.py')
    # on peut return l'id du model output