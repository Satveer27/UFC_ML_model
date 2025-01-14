from fastapi import FastAPI, HTTPException, Body
import numpy as np
import joblib
from typing import Dict
from utils.clean import clean_pds, cleaned_for_ml, combine_fighters, get_fighter_id, present_result
from pydantic import BaseModel

class PredictionResult(BaseModel):
    winner: str
    time: str
    round: int
    method: str

# After running the model from ufc_model.py then use the joblib file here
model = joblib.load('ufc_model.joblib')

app = FastAPI(
    title="UFC API",
    description="An API for predicting UFC fight outcomes. This is the documentation and explanation of the API will be provided below with examples on how to use such API",
    version="1.0.0"
)

@app.get('/',
        summary="Welcome to the UFC API",
        description=(
        "This is the root endpoint of the UFC API. "
        "Use this API to predict fight outcomes"
        "Explore detailed endpoints at `/docs`"))
async def direct_root():
    return {"message": "Welcome to the UFC API. Visit /docs for more details."}

@app.post(
    "/predict",
    summary="Predict Fight Outcome",
    description="""
    Predict the winner of a UFC fight between two fighters. 
    Provide the IDs of two fighters in the **JSON body**, and the model will return the predicted winner, time of victory, knockout method, and round.
    """,
    response_model=PredictionResult,
    responses={
        200: {
            "description": "Prediction Successful",
            "content": {
                "application/json": {
                    "example": {
                        "winner": "fighter1_id",
                        "time": "00:03:45",
                        "round": 3,
                        "method": "KO/TKO"
                    }
                }
            },
        },
        400: {"description": "Bad Request - Missing Fighter IDs"},
        500: {"description": "Internal Server Error"},
    })
def direct_predict(data: Dict = Body(
        ...,
        example={
            "fighter1": "fighter1_id",
            "fighter2": "fighter2_id",
        },
    )
    ):
    # Predicts which fighter is going to win, in the headers pass the fighter id and then you will get the fighters

    fighter1 = data.get('fighter1')
    fighter2 = data.get('fighter2')

    if not fighter1 or not fighter2:
        raise HTTPException(status_code=400, detail="Both fighter1 and fighter2 must be provided")
    
    try:
        fighter1_data = get_fighter_id(fighter1)
        fighter2_data = get_fighter_id(fighter2)

        if not fighter1_data:
            raise HTTPException(status_code=400, detail=f"Could not fetch data for fighter1: {fighter1}")
        
        if not fighter2_data:
            raise HTTPException(status_code=400, detail=f"Could not fetch data for fighter2: {fighter2}")
    
        combined_pd = combine_fighters(fighter1_data, fighter2_data)

        cleaned_pd_for_ml = cleaned_for_ml(combined_pd)

        cleaned_pds_ml = clean_pds(cleaned_pd_for_ml)

        prediction = model.predict(cleaned_pds_ml)

        cleaned_prediction = present_result(prediction, fighter1, fighter2)
        
        return PredictionResult(
            winner=cleaned_prediction[0],
            time=str(cleaned_prediction[1]),
            round=int(cleaned_prediction[2]),
            method=cleaned_prediction[3]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    


