import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib
import shap

# Import model 
loaded_model = joblib.load('./best_Random Forest_2024-07-26.joblib')
shap_values_global = joblib.load('./SHAP/shap_values.joblib')

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
@app.get("/acceuil")
def great():
   return {"message":"Bonjour"}

# Endpoint pour récupérer les ID_CLIENT
@app.get("/CLIENTS")
def Liste_client():
    # Renvoie tous les ID_CLIENT disponibles dans le DataFrame
    ID_CLIENT = sorted(df_api.index.tolist())
    return {"ID_CLIENT": ID_CLIENT}

@app.get("/INFO_CLIENTS")
def info_client(ID_CLIENT: int):
   INFO_CLIENT = Base_client.loc[[ID_CLIENT]]
   INFO_CLIENT_dict = INFO_CLIENT.to_dict()
   return INFO_CLIENT_dict

@app.get("/predict")
def predict(ID_CLIENT) :
   ID_CLIENT = int(ID_CLIENT)
   # Vérifier si l'ID_CLIENT existe dans le DataFrame
   if ID_CLIENT in df_api.index:
      # Extraire les données du client
      client_data = df_api.loc[ID_CLIENT].values.reshape(1, -1)
      
      # Tester le modèle sur ce ID_CLIENT
      prediction = loaded_model.predict_proba(client_data)
      prediction = prediction[0][1]
      # Afficher la prédiction
      return {'prediction': prediction}
   else:
      return 'Manquant'
   
@app.get("/SHAP_GLOBAL")
def shap_global() :
   # Faire le shap par classe
   shap_values_class_1_global = shap_values_global[..., 1]
   # Le téléchargement des shap c'est fait sur une base réduite pour la taille des données
   # Possibilité d'intégrer une image à la place mais modification max_display impossible
   
   return {
        'shap_values_class_1_global': shap_values_class_1_global.tolist(),  # Conversion en liste JSON-compatible
        'df_api': df_api.to_dict(orient="records"),  # Conversion du DataFrame en dict
        'df_api_columns': df_api.columns.tolist()  # Conversion des colonnes en liste
    }

@app.get("/shap_individual")
def shap_individual(ID_CLIENT) :
   # Selection personne
   observation = df_api.loc[[ID_CLIENT]]
   # Shap_values individue
   shap_values_ind = explainer.shap_values(observation)
   # Select SHAP values for the first output 
   shap_values_class_1_ind = shap_values_ind[..., 1] 

   return {
        "shap_values_class_1": shap_values_class_1_ind,
        "observation": observation,
        "columns": observation.columns
    }
  
if __name__ == "__main__":
   uvicorn.run("Prediction_api:app", host="127.0.0.1", port=8000, reload=True)