# Machine learning UFC fighter model

This is a project that trains a machine learning model based on data taken via a ETL for ufc fighters. This model predicts which fighter won given 2 fighters as well as the time taken, round and method of KO for the predicted result. The project utilises pandas for preprocessing and cleaning the data. The project also utilises scikit learn to use various models such as random forest in my case. 

This project is a extension of the ETL i created and to run this project you must have a azure database with ufc fighter data, ufc fights, ufc events and ufc fight details. The backend of the model is developed with fast API so that it can be deployed and can be used. To use the API itself you will need to run the ML model yourself and get the job lib file. Then write the file name in the model variable inside the api file. 






