from fastapi import FastAPI
import numpy as np
import joblib
from typing import Dict
from utils.clean import clean_pds, cleaned_for_ml, combine_fighters, get_fighter_id, present_result
from utils.db_connec import connect_to_database
import requests

# After running the model from ufc_model.py then use the joblib file here
model = joblib.load('ufc_model.joblib')

app = FastAPI()

@app.get('/')
def direct_root():
    return {'message': 'ml model connected'}

@app.post('/predict')
def direct_predict(data: Dict):
    # Predicts which fighter is going to win, in the headers pass the fighter id and then you will get the fighters

    fighter1 = data.get('fighter1')
    fighter2 = data.get('fighter2')

    if not fighter1 or not fighter2:
        return{"error": "Both fighter1 and fighter2 must be provided"}
    
    try:
        fighter1_data = get_fighter_id(fighter1)
        fighter2_data = get_fighter_id(fighter2)

        if fighter1_data == False:
            return {"error": f"Could not fetch data for fighter1: {fighter1}"}
        
        if fighter2_data == False:
            return {"error": f"Could not fetch data for fighter1: {fighter2}"}
    
        combined_pd = combine_fighters(fighter1_data, fighter2_data)

        cleaned_pd_for_ml = cleaned_for_ml(combined_pd)

        cleaned_pds_ml = clean_pds(cleaned_pd_for_ml)

        prediction = model.predict(cleaned_pds_ml)

        cleaned_prediction = present_result(prediction, fighter1, fighter2)

        result = {
        "winner": cleaned_prediction[0], 
        "time": str(cleaned_prediction[1]),  
        "round": cleaned_prediction[2],
        "method": cleaned_prediction[3]
        }
        
        return{"message": result}
    
    except Exception as e:
        return{"error": f'{e}'}
    
    