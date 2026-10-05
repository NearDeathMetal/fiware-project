#Bibliotecas FastAPI
from fastapi import FastAPI, HTTPException, Request
import httpx,json
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

#Third-party imports
import numpy as np
from joblib import load
import pandas as pd
import os

#Modelos Pydantic para input do modelo
class InputData(BaseModel):
    e2sp_co: float
    e2sp_co_we: float
    e2sp_co_ae: float
    e2sp_temp: float
    pin_umid: float

class PredictionResult(BaseModel):
    CO_previsto: float 

app = FastAPI(title="API para correção de dados da Estação Envcity")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


#Carregando modelo
model_path = os.path.join(os.path.dirname(__file__), 'RF_Regressor.joblib')
modelo = load('RF_Regressor.joblib') 


#ORION_BASE_URL = os.getenv("ORION_BASE_URL", "http://orion:1026")
DOCKER_HOST= "orion"
LOCAL_HOST = "localhost"

ORION_ENTITIES_URL = f"http://{DOCKER_HOST}:1026/v2/entities/"
ORION_SUBSCRIPTIONS_URL = f"http://{DOCKER_HOST}:1026/v2/subscriptions"
ORION_VERSION_URL = f"http://{DOCKER_HOST}:1026/version/"
API_SERVICE_NAME = os.getenv("API_SERVICE_NAME", "ml-api")


FIWARE_SERVICE = "openiot"
FIWARE_SERVICEPATH = "/airQuality"

ORION_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "fiware-service": FIWARE_SERVICE,
    "fiware-servicepath": FIWARE_SERVICEPATH
}

ORION_GET_HEADERS = {
    "Accept": "application/json",
    "fiware-service": "openiot",
    "fiware-servicepath": "/airQuality"
}


#Conexão ORION erro
def erro_orion(e: httpx.HTTPStatusError):
    return JSONResponse(
        content={"error": f"Falha na requisição ao Orion: {str(e)}"},
        status_code=e.response.status_code
    )
#erro interno
def erro_interno(e: Exception):
    return JSONResponse(
        content={"error": f"Erro inesperado: {str(e)}"},
        status_code=500
    )

#para fazer GET request com o httpx
async def async_request(url: str, headers: dict):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            print(url)
            response.raise_for_status()
                   
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                print(response.json()) 
                return response.json()
            else:
                return response.text
             
    except httpx.HTTPStatusError as e:
        return erro_orion(e)
    except Exception as e:
        return erro_interno(e)
    

@app.get("/",tags=['Welcome!!'])
def welcome():
    return {"status_code": "200",
            "description": "IT WORKS!"}

#lista entidade do orion CB
@app.get("/orion/entities", summary="📄 Listar Entidades do Orion",tags=['Orion CB operations'])
async def listar_entidades_orion():
    try:
        entidades = await async_request(ORION_ENTITIES_URL, headers=ORION_GET_HEADERS)
        print(entidades)
        return JSONResponse(content=entidades)
    except httpx.HTTPStatusError as e:
        return erro_orion(e)
    except Exception as e:
        return erro_interno(e)

#testa conexão
@app.get("/orion/status", summary="🔗 Testar Conexão com Orion",tags=['Orion CB operations'])
async def testar_conexao():
    try:
        async with httpx.AsyncClient() as client:
                response = await client.get(ORION_VERSION_URL)
                response.raise_for_status()
                return JSONResponse(response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as e:
        return erro_orion(e)


#request de predição com base no modelo carregado
@app.post("/prediction", response_model=PredictionResult, tags=['Prediction'])
async def calculate_prediction(data: InputData):
    
    #Constrói o vetor de entrada
    entrada = pd.DataFrame([[data.e2sp_co, data.e2sp_co_we, data.e2sp_co_ae, data.e2sp_temp, data.pin_umid]],columns=['e2sp_co', 'e2sp_co_we', 'e2sp_co_ae', 
    'e2sp_temp', 'pin_umid'])  # Use os mesmos nomes do treino # 1 amostra, 5 features calculados pelo numpy

    # Realiza a predição
    resultado = modelo.predict(entrada)

    return PredictionResult(CO_previsto=resultado[0].round(6))

#cria inscrição orion fazer o POST quando entidade SensorCvel atualiza
@app.post("/orion/subscribe", summary="📡 Notificar quando Orion CB atualizar", tags=['Notification'])
async def orion_subscripton():
    ORION_SUBSCRIPTION_URL = f"http://{DOCKER_HOST}:1026/v2/subscriptions"

    PREDICTION_ENDPOINT = f"http://{API_SERVICE_NAME}:8000/notifyCO"
    
    headers = {"Content-Type": "application/json","Content-Type": "application/json",
    "fiware-service": FIWARE_SERVICE,
    "fiware-servicepath": FIWARE_SERVICEPATH}
    payload = {
        "description": "Subscribe to LoraDevice updates",
        "subject": {
            "entities": [
                {
                    "idPattern": ".*",
                    "type": "LoraDevice"
                    #para todos "idPattern": ".*"  e para especifico:"id": "SensorCvel"
                }
            ],
            "condition": {
                "attrs": ["Best_CO", "CO_WE", "CO_AE", "Temperatura", "Umidade"]
            }
        },
        "notification": {
            "http": {
                "url": PREDICTION_ENDPOINT#<URL_DA_API>
            },
            "attrs": ["Best_CO", "CO_WE", "CO_AE", "Temperatura", "Umidade"]
        },
        "throttling": 1
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(ORION_SUBSCRIPTION_URL, json=payload, headers=headers)
            response.raise_for_status()
            if response.status_code == 201:
                return {"status": "Subscription criada com sucesso!"}
            return {"status": "Subscription resposta recebida", "content": response.text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/notifyCO",
    summary="🤖 Recebe atualização Orion e calcula CO corrigido",
    tags=['Notification']
)
async def co_prediction(notification: dict):

    print("\n===== NOTIFICAÇÃO ORION =====")
    print(json.dumps(notification, indent=4))


    try:

        # Orion envia lista de entidades
        entity = notification["data"][0]


        entity_id = entity["id"]


        # Recupera valores enviados pelo Orion

        e2sp_co = entity.get(
            "Best_CO",
            {}
        ).get(
            "value",
            0.0
        )


        e2sp_co_we = entity.get(
            "CO_WE",
            {}
        ).get(
            "value",
            0.0
        )


        e2sp_co_ae = entity.get(
            "CO_AE",
            {}
        ).get(
            "value",
            0.0
        )


        e2sp_temp = entity.get(
            "Temperatura",
            {}
        ).get(
            "value",
            0.0
        )


        pin_umid = entity.get(
            "Umidade",
            {}
        ).get(
            "value",
            0.0
        )



        entrada = pd.DataFrame(
            [[
                e2sp_co,
                e2sp_co_we,
                e2sp_co_ae,
                e2sp_temp,
                pin_umid
            ]],
            columns=[
                'e2sp_co',
                'e2sp_co_we',
                'e2sp_co_ae',
                'e2sp_temp',
                'pin_umid'
            ]
        )


        resultado = modelo.predict(entrada)


        co_corrigido = round(
            float(resultado[0]),
            6
        )


        print(
            f"CO corrigido calculado: {co_corrigido}"
        )



        # Atualiza Orion

        url = (
            f"{ORION_ENTITIES_URL}"
            f"{entity_id}/attrs"
        )


        payload = {

            "CO_Corrigido": {

                "type":"Number",

                "value":co_corrigido

            }

        }



        async with httpx.AsyncClient() as client:

            response = await client.patch(
                url,
                json=payload,
                headers=ORION_HEADERS
            )

            if response.status_code == 422:
                print(
                    "CO_Corrigido não existe. "
                    "Criando atributo no Orion..."
                )

                response = await client.post(
                    url,
                    json=payload,
                    headers=ORION_HEADERS
                )

            response.raise_for_status()



        return {

            "status":"OK",

            "entity":entity_id,

            "CO_Corrigido":co_corrigido

        }



    except Exception as e:

        print(
            "Erro no processamento:",
            e
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )
