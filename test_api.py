import unittest
import requests
import joblib
import pandas as pd

class TestApi(unittest.TestCase):
    
    # Test pour vérifier que le modèle est bien chargé
    def test_model_loading(self):
        model_path = './best_Random Forest_2024-07-26.joblib'
        try:
            loaded_model = joblib.load(model_path)
            self.assertIsNotNone(loaded_model, "Le modèle n'a pas été chargé correctement.")
            print(f"Le modèle a bien été chargé.")
        except FileNotFoundError:
            self.fail(f"Le fichier du modèle {model_path} est introuvable.")


    # Test pour vérifier que les valeurs SHAP sont bien chargées
    def test_shap_loading(self):
        shap_path = './SHAP/shap_values.joblib'
        try:
            shap_values_global = joblib.load(shap_path)
            self.assertIsNotNone(shap_values_global, "Les valeurs SHAP n'ont pas été chargées correctement.")
            print(f"Les valeurs SHAP ont bien été chargées")
        except FileNotFoundError:
            self.fail(f"Le fichier SHAP {shap_path} est introuvable.")

    # Test pour vérifier le chargement de la base de données X_test
    def test_loading_X_test(self):
        csv_path = './X_test.csv'
        try:
            df_api = pd.read_csv(csv_path, index_col="ID_CLIENT")
            df_api.index = df_api.index.astype(int)
            self.assertFalse(df_api.empty, "Le DataFrame X_test est vide.")
            print(f"Le fichier X_test a été correctement chargé")
        except FileNotFoundError:
            self.fail(f"Le fichier X_test {csv_path} est introuvable.")

    
    # Test pour vérifier le chargement de la base de données Base_client
    def test_loading_Base_client(self):
        csv_path_base_client = './Base_client.csv'
        try:
            Base_client = pd.read_csv(csv_path_base_client, index_col='Unnamed: 0')
            Base_client.index = Base_client.index.astype(int)
            self.assertFalse(Base_client.empty, "Le DataFrame Base_client est vide.")
            print(f"Le fichier Base_client a été correctement chargé") 
        except FileNotFoundError:
            self.fail(f"Le fichier base_client est introuvable.") # Ne pas donner l'adresse de ce document

    # Test pour vérifier si le prédict fonctionne bien
    def test_get_predict_client(self):
        try :
            # Load model
            model_path = './best_Random Forest_2024-07-26.joblib'
            loaded_model = joblib.load(model_path)
            # Load client data
            csv_path = './X_test.csv'
            df_api = pd.read_csv(csv_path, index_col="ID_CLIENT")
            df_api.index = df_api.index.astype(int)
            # Predict sur le premier
            client_data = df_api.iloc[[0]]
            prediction = loaded_model.predict_proba(client_data)
            self.assertIsNotNone(prediction, "La prédiction ne fonctionne pas correctement.")
            print(f"La prédiction fonctionne.")
        except FileNotFoundError:
            self.fail(f"La prédiction n'a pas chargé correctement")
        


if __name__ == '__main__':
    unittest.main()

