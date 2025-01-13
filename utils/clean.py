import pandas as pd
from datetime import timedelta

def combine_fighters(fighter1, fighter2):
    try:
        fighter1_data_pd = pd.DataFrame([fighter1])
        fighter2_data_pd = pd.DataFrame([fighter2]).add_prefix("fighter2_")

        return pd.concat([fighter1_data_pd, fighter2_data_pd], axis=1)
    
    except:
        return False

def cleaned_for_ml(combined_pd):
    relevant_columns = combined_pd[[
    # Fighter 1 stats
    'height', 'weight', 'reach', 'stance',
    'splm', 'str_acc', 'sapm', 'str_def',
    'td_avg', 'td_acc', 'td_def', 'sub_avg',
    'wins', 'losses', 'draws', 'no-contest',
    
    # Fighter 2 stats (suffix "fighter_2")
    'fighter2_height', 'fighter2_weight', 'fighter2_reach', 'fighter2_stance',
    'fighter2_splm', 'fighter2_str_acc', 'fighter2_sapm', 'fighter2_str_def',
    'fighter2_td_avg', 'fighter2_td_acc', 'fighter2_td_def', 'fighter2_sub_avg',
    'fighter2_wins', 'fighter2_losses', 'fighter2_draws', 'fighter2_no-contest',

    ]]

    return relevant_columns

def clean_pds(cleaned_pds_ml):

    STANCE_COLUMNS_F1 = [
    'stance_f1_', 'stance_f1_Open Stance', 'stance_f1_Orthodox',
    'stance_f1_Sideways', 'stance_f1_Southpaw', 'stance_f1_Switch'
    
    ]

    STANCE_COLUMNS_F2 = [
        'stance_f2_', 'stance_f2_Open Stance', 'stance_f2_Orthodox',
        'stance_f2_Sideways', 'stance_f2_Southpaw', 'stance_f2_Switch'
    ]

    cleaned_for_ml_copy = cleaned_pds_ml.copy()

    cleaned_for_ml_copy['height'] = cleaned_for_ml_copy['height'].apply(height_to_cm)
    cleaned_for_ml_copy['fighter2_height'] = cleaned_for_ml_copy['fighter2_height'].apply(height_to_cm)

    cleaned_for_ml_copy["weight"] = cleaned_for_ml_copy['weight'].apply(getWeight)
    cleaned_for_ml_copy["fighter2_weight"] = cleaned_for_ml_copy['fighter2_weight'].apply(getWeight)

    cleaned_for_ml_copy["reach"] = cleaned_for_ml_copy['reach'].apply(getReach)
    cleaned_for_ml_copy["fighter2_reach"] = cleaned_for_ml_copy['fighter2_reach'].apply(getReach)

    fighter1_stance = pd.get_dummies(cleaned_for_ml_copy['stance'], prefix='stance_f1').astype(int)
    fighter2_stance = pd.get_dummies(cleaned_for_ml_copy['fighter2_stance'], prefix='stance_f2').astype(int)

    for col in STANCE_COLUMNS_F1:
        if col not in fighter1_stance:
            fighter1_stance[col] = 0

    for col in STANCE_COLUMNS_F2:
        if col not in fighter2_stance:
            fighter2_stance[col] = 0
    
    fighter1_stance = fighter1_stance[STANCE_COLUMNS_F1]
    fighter2_stance = fighter2_stance[STANCE_COLUMNS_F2]

    cleaned_for_ml_copy = cleaned_for_ml_copy.drop(columns=['stance', 'fighter2_stance'])

    cleaned_for_ml_copy = pd.concat([cleaned_for_ml_copy, fighter1_stance, fighter2_stance], axis=1)

    cleaned_for_ml_copy['str_acc'] = cleaned_for_ml_copy['str_acc'].apply(to_int)
    cleaned_for_ml_copy['str_def'] = cleaned_for_ml_copy['str_def'].apply(to_int)
    cleaned_for_ml_copy['td_acc'] = cleaned_for_ml_copy['td_acc'].apply(to_int)
    cleaned_for_ml_copy['td_def'] = cleaned_for_ml_copy['td_def'].apply(to_int)

    cleaned_for_ml_copy['fighter2_str_acc'] = cleaned_for_ml_copy['fighter2_str_acc'].apply(to_int)
    cleaned_for_ml_copy['fighter2_str_def'] = cleaned_for_ml_copy['fighter2_str_def'].apply(to_int)
    cleaned_for_ml_copy['fighter2_td_acc'] = cleaned_for_ml_copy['fighter2_td_acc'].apply(to_int)
    cleaned_for_ml_copy['fighter2_td_def'] = cleaned_for_ml_copy['fighter2_td_def'].apply(to_int)

    # column mapping
    column_mapping = {
    "fighter2_height": "height_fighter_2",
    "fighter2_weight": "weight_fighter_2",
    "fighter2_reach": "reach_fighter_2",
    "fighter2_splm": "splm_fighter_2",
    "fighter2_str_acc": "str_acc_fighter_2",
    "fighter2_sapm": "sapm_fighter_2",
    "fighter2_str_def": "str_def_fighter_2",
    "fighter2_td_avg": "td_avg_fighter_2",
    "fighter2_td_acc": "td_acc_fighter_2",
    "fighter2_td_def": "td_def_fighter_2",
    "fighter2_sub_avg": "sub_avg_fighter_2",
    "fighter2_wins": "wins_fighter_2",
    "fighter2_losses": "losses_fighter_2",
    "fighter2_draws": "draws_fighter_2",
    "fighter2_no-contest": "no-contest_fighter_2"
    }

    cleaned_for_ml_copy.rename(columns= column_mapping, inplace = True)

    return cleaned_for_ml_copy


def determine_winner(row):
    try:
        fight_winner = row['fight_winner']
        fighter1_id = row['fighter1_fight_id']
        if(fight_winner is not None):
            if(fight_winner == fighter1_id):
                return 0
            else:
                return 1
        else:
            return -1
        
    except Exception as e:
        return -1

def height_to_cm(height_str):
    try:
        if isinstance(height_str, (int, float)):
            return float(height_str)
        
        feet = height_str.strip("'")[0]
        feet = int(feet)
        try:
            inches = height_str.strip("'")[3]
            inches = int(inches)
        except:

            inches = 0
        
        total_inches = (feet*12) + inches
        total_cm = total_inches * 2.54
        return total_cm
    except:
        return 0
    
def getWeight(weight_str):
    try:
        if(isinstance(weight_str, (int, float))):
            return weight_str
        
        return int(weight_str.split(" lbs.")[0])
    except Exception as e:
        print(f'error {e} for {weight_str}')
        return 0    

def getReach(reach_str):
    try:
        if(isinstance(reach_str, (int, float))):
            return reach_str
        
        return int(reach_str.split('"')[0])
    except Exception as e:
        print(f'error {e} for {reach_str}')
        return 0
    
def convert_stance(array, column, prefix_name):
    stance_f1 = pd.get_dummies(array[column], prefix=prefix_name)
    stance_f1 = stance_f1.astype(int)
    array = pd.concat([array, stance_f1], axis=1)
    array.drop(columns=[column], inplace=True)
    return array

def to_int(value):
    try:
        return int(value)  # Attempt to convert to integer
    except:
        return 0
    
def convert_to_time(value):
    try:
        hour = 3600 * value.hour
        minutes = value.minute * 60
        total_sec = hour + minutes + value.second
        return total_sec
    except:
        return 0
    
def convert_knockout(value):
    methods_of_knockout = {0: 'U-DEC', 1: 'SUB', 2: 'KO/TKO', 3: 'S-DEC', 4: 'M-DEC', 5: 'DQ'}
    return methods_of_knockout.get(value, -1)

def present_result(data, fighter1, fighter2):
    final = []

    if(data[0][0] == 0):
        final.append(fighter1)
    elif(data[0][0] == 1):
        final.append(fighter2)
    else:
        final.append("--")

    time = data[0][1]
    fight_time = timedelta(seconds=time)
    final.append(fight_time)
    round = data[0][2]
    final.append(round)
    method = convert_knockout(data[0][3])
    final.append(method)

    return final

