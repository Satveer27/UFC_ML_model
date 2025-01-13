from utils.clean import convert_knockout, convert_stance, convert_to_time, determine_winner, getReach, getWeight, height_to_cm, to_int
from utils.db_connec import connect_to_database
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, mean_absolute_error

#This is a ufc model to predict any upcoming ufc fights and the model is trained on up to date UFC data
# UFC data is scraped and stored in a database via an ETL and the code for this is uploaded in a different repo

def get_data():
    conn = connect_to_database()
    sql_query = """
    SELECT *
    FROM ufc_fighters
    """

    sql_query_2 = """
    SELECT *
    FROM fights
    """
    sql_query_3 = """
    SELECT *
    FROM fight_details
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        fighters_columns = [desc[0] for desc in cursor.description]  
        all_fighters = cursor.fetchall()
        all_fighters = [list(row) for row in all_fighters]
        
        cursor.execute(sql_query_2)
        fight_columns = [column[0] for column in cursor.description]
        all_fights = cursor.fetchall()
        all_fights = [list(row) for row in all_fights]

        cursor.execute(sql_query_3)
        fight_det_columns = [column[0] for column in cursor.description]
        all_fights_details = cursor.fetchall()
        all_fights_details = [list(row) for row in all_fights_details]
        
        all_fighters_pandas = pd.DataFrame(all_fighters, columns=fighters_columns)
        all_fights_pandas = pd.DataFrame(all_fights, columns=fight_columns)
        all_fights_details_pandas = pd.DataFrame(all_fights_details, columns=fight_det_columns)

        conn.close()

        return all_fighters_pandas, all_fights_pandas, all_fights_details_pandas
    except Exception as e:
        raise(e)

# if u want to see how the model performed you can call this
def make_predictions(model, testing_sample, actual_result):
    # test model, get the classification report to view if relevant output had good precision
    y_pred = model.predict(testing_sample)

    winner_pred = y_pred[:, 0] 
    time_pred = y_pred[:, 1]    
    round_pred = y_pred[:, 2]   

    print(classification_report(actual_result['cleaned_fight_winner'], winner_pred, zero_division=0))
    print(mean_absolute_error(actual_result['time'], time_pred))
    print(mean_absolute_error(actual_result['round'], round_pred))

all_fighters_pds, all_fights_pds, all_fights_details_pandas = get_data()

#clean the data
combined_fights = all_fights_details_pandas.merge(
    all_fighters_pds,
    left_on="fighter1_fight_id",
    right_on="fighter_id",
    how="left",
    suffixes=("", "_fighter_1")
)

combined_fights = combined_fights.merge(
    all_fighters_pds,
    left_on="fighter2_fight_id",
    right_on="fighter_id",
    how="left",
    suffixes=("", "_fighter_2") 
)

combined_fights = combined_fights.merge(
    all_fights_pds[['fight_id', 'methodOfKnockout', 'round', 'time']],
    on = 'fight_id',
    how='left'
)
    
combined_fights["cleaned_fight_winner"] = combined_fights.apply(determine_winner, axis=1)

#cleaning process
relevant_columns = combined_fights[[
    # Fighter 1 stats
    'height', 'weight', 'reach', 'stance',
    'splm', 'str_acc', 'sapm', 'str_def',
    'td_avg', 'td_acc', 'td_def', 'sub_avg',
    'wins', 'losses', 'draws', 'no-contest',
    
    # Fighter 2 stats (suffix "_fighter_2")
    'height_fighter_2', 'weight_fighter_2', 'reach_fighter_2', 'stance_fighter_2',
    'splm_fighter_2', 'str_acc_fighter_2', 'sapm_fighter_2', 'str_def_fighter_2',
    'td_avg_fighter_2', 'td_acc_fighter_2', 'td_def_fighter_2', 'sub_avg_fighter_2',
    'wins_fighter_2', 'losses_fighter_2', 'draws_fighter_2', 'no-contest_fighter_2',
    
    # Fight context
    'cleaned_fight_winner', "methodOfKnockout", "round", "time"
]]

#preprocess data
relevant_columns = relevant_columns.copy()

relevant_columns['height'] = relevant_columns['height'].apply(height_to_cm)
relevant_columns['height_fighter_2'] = relevant_columns['height_fighter_2'].apply(height_to_cm)

relevant_columns["weight"] = relevant_columns['weight'].apply(getWeight)
relevant_columns["weight_fighter_2"] = relevant_columns['weight_fighter_2'].apply(getWeight)

relevant_columns["reach"] = relevant_columns['reach'].apply(getReach)
relevant_columns["reach_fighter_2"] = relevant_columns['reach_fighter_2'].apply(getReach)

relevant_columns = convert_stance(relevant_columns, "stance", "stance_f1")
relevant_columns = convert_stance(relevant_columns, "stance_fighter_2", "stance_f2")

relevant_columns['str_acc'] = relevant_columns['str_acc'].apply(to_int)
relevant_columns['str_def'] = relevant_columns['str_def'].apply(to_int)
relevant_columns['td_acc'] = relevant_columns['td_acc'].apply(to_int)
relevant_columns['td_def'] = relevant_columns['td_def'].apply(to_int)
relevant_columns['str_acc_fighter_2'] = relevant_columns['str_acc_fighter_2'].apply(to_int)
relevant_columns['str_def_fighter_2'] = relevant_columns['str_def_fighter_2'].apply(to_int)
relevant_columns['td_acc_fighter_2'] = relevant_columns['td_acc_fighter_2'].apply(to_int)
relevant_columns['td_def_fighter_2'] = relevant_columns['td_def_fighter_2'].apply(to_int)

relevant_columns['round'] = relevant_columns['round'].fillna(0)
relevant_columns['time'] = relevant_columns['time'].apply(convert_to_time)
relevant_columns['methodOfKnockout'] = relevant_columns['methodOfKnockout'].apply(convert_knockout)

#Splitting dataset for training and testing. 80% training and 20% testing
y = relevant_columns[['cleaned_fight_winner', 'time', 'round', 'methodOfKnockout']]
X = relevant_columns.drop(columns=['cleaned_fight_winner', 'time', 'round', 'methodOfKnockout'])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 ,test_size=0.2, random_state=42)

# Training model on random forest to predict multiple output
model = MultiOutputRegressor(RandomForestClassifier(n_estimators=100, min_samples_split=40 ,random_state=42))
model.fit(X_train, y_train)

# Remove if you dont want to see how precise model is
make_predictions(model, X_test, y_test)

#Save model
joblib.dump(model, 'ufc_model.joblib')
