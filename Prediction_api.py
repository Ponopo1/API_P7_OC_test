import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import shap
from functools import lru_cache

# Import model 
loaded_model = joblib.load('./best_Random Forest_2024-07-26.joblib')
shap_values_global = joblib.load('./SHAP/shap_values.joblib')

@lru_cache(maxsize=1)
def load_shap_values():
    return joblib.load('./SHAP/shap_values.joblib')

# 2. Lire le fichier CSV dans un DataFrame en ignorant la colonne d'index si nécessaire
csv_path = './X_test.csv'
df_api= pd.read_csv(csv_path, index_col="ID_CLIENT")
df_api.index = df_api.index.astype(int)

# Base_client import
csv_path_base_client = './Base_client.csv'
Base_client= pd.read_csv(csv_path_base_client, index_col='Unnamed: 0')
Base_client.index = Base_client.index.astype(int)

explainer = shap.TreeExplainer(loaded_model)

# Instance API
app = FastAPI()

def liste_client_df_api() :
   # Renvoie tous les ID_CLIENT disponibles dans le DataFrame
    ID_CLIENT = sorted(df_api.index.tolist())
    return ID_CLIENT

# Endpoint pour récupérer les ID_CLIENT
@app.get("/CLIENTS")
def Liste_client():
    # Renvoie tous les ID_CLIENT disponibles dans le DataFrame
    ID_CLIENT = liste_client_df_api()
    return {"ID_CLIENT": ID_CLIENT}

def liste_client_base_client(ID_CLIENT) :
   INFO_CLIENT = Base_client.loc[[ID_CLIENT]]
   INFO_CLIENT_dict = INFO_CLIENT.to_dict()
   return INFO_CLIENT_dict

@app.get("/INFO_CLIENTS")
def info_client(ID_CLIENT: int):
   INFO_CLIENT = liste_client_base_client(ID_CLIENT)
   return INFO_CLIENT

@app.post("/INFO_CLIENTS_GLOBAL")
def info_client_global():
   # Sélectionner uniquement les colonnes numériques pour diminuer le temps de réponse
    base_client_numeric = Base_client.select_dtypes(include=['number']).copy()
    # Ajouter une colonne 'ID' basée sur l'index
    base_client_numeric['ID'] = Base_client.index
    # Calcul proba
    df_proba = pd.DataFrame({
   'ID': df_api.index,  # Utiliser l'index de df_api comme ID
   'proba': loaded_model.predict_proba(df_api)[:,1]})
    df_proba.set_index('ID', inplace=True)
    # intégration de la proba 
    base_client_numeric_prob = base_client_numeric.join(df_proba[['proba']], how='inner')
    # Convertir en dictionnaire
    info_client_global_dict = base_client_numeric_prob.to_dict(orient="records")
    
    return info_client_global_dict

# Définition pour le test
def predict_from_id_client(id_client) :
   id_client = int(id_client)
   # Vérifier si l'ID_CLIENT existe dans le DataFrame
   if id_client in df_api.index:
      # Extraire les données du client
      client_data = df_api.loc[id_client].values.reshape(1, -1)
      
      # Tester le modèle sur ce ID_CLIENT
      prediction = loaded_model.predict_proba(client_data)
      prediction = prediction[0][1]
      # Afficher la prédiction
      return prediction
   else:
      return None
@app.get("/predict")
def predict(id_client) :
   prediction = predict_from_id_client(id_client) 
   if prediction is not None : 
      return {'ID_CLIENT': id_client,
              'prediction' : prediction}
   else :
      return 'Manquant'
   
   
@app.get("/SHAP_GLOBAL")
def shap_global():
   importances = loaded_model.feature_importances_
   std = np.std([tree.feature_importances_ for tree in loaded_model.estimators_], axis=0)
   df_api_columns = df_api.columns.tolist()

   # Convertissez les importances en un dictionnaire
   return {'features': df_api_columns,
      'importances': importances.tolist(),
      'std_dev': std.tolist()}   

@app.get("/shap_individual")
def shap_individual(ID_CLIENT :int) :
   # Selection personne
   observation = df_api.loc[[ID_CLIENT]]
   observation_dict = observation.to_dict()
   observation_list = observation.columns.tolist()
   # Shap_values individue
   shap_values_ind = explainer.shap_values(observation)
   # Select SHAP values for the first output 
   shap_values_class_1_ind = shap_values_ind[..., 1] 
   shap_values_class_1_ind_list = shap_values_class_1_ind.tolist()
   return {
        "shap_values_class_1": shap_values_class_1_ind_list,
        "observation": observation_dict,
        "columns": observation_list}

if __name__ == "__main__":
   uvicorn.run("Prediction_api:app", host="127.0.0.1", port=8000, reload=True)