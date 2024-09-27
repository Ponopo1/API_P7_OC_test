import unittest
import requests
import joblib
import pandas as pd
from Prediction_api import predict_from_id_client, liste_client_df_api,liste_client_base_client

class TestApi(unittest.TestCase):
    
    # Test pour vérifier le chargement de la base de données x_test (df_api)
    def test_liste_client_df_api(self):
        # Given
        client_list = liste_client_df_api()
        # When
        computed_result = len(set(client_list))
        # Then 
        assert computed_result == 2332 # Nb à changer si changement de données
        print('Nous avons chargé le bon fichier X_test')

    # Test pour vérifier si le prédict fonctionne bien
    def test_predict_from_id_client_when_id_client_does_not_exist(self):
        # Given 
        non_existent_id_client = 10000000000
        # When        
        computed_result = predict_from_id_client(non_existent_id_client)
        # Then
        assert computed_result is None
        print('Le faux client n\'existe pas')

    def test_predict_from_id_client_when_id_client_does_exist(self):
        # Given 
        existent_id_client = 49386
        # When        
        computed_result = predict_from_id_client(existent_id_client)
        # Then
        expected_result = 0.289
        assert expected_result-computed_result < 0.001
        print(f"Le resultat prédict pour {existent_id_client} est bon") 
    
    # Test de nos données générales
    def test_info_client_when_id_does_not_exist(self):
        # Given 
        non_existent_id_client = 10000000000
        # When        
        computed_result = liste_client_base_client(non_existent_id_client)
        # Then
        assert computed_result is None
        print('Pas de fausse donnée chargée')
    
    def test_info_client_when_id_does_not_exist(self):
        # Given 
        existent_id_client = 49386
        # When        
        computed_result = liste_client_base_client(existent_id_client)
        # Then
        expected_result = {
            'AMT_CREDIT': {49386: 450000.0},
            'AMT_ANNUITY': {49386: 20979.0},
            'AMT_GOODS_PRICE': {49386: 450000.0},
            'DAYS_BIRTH': {49386: -14351},
            'CNT_CHILDREN': {49386: 0},
            'DAYS_EMPLOYED': {49386: -4551.0},
            'NAME_EDUCATION_TYPE': {49386: 'Incomplete higher'},
            'NAME_CONTRACT_TYPE': {49386: 'Cash loans'},
            'NAME_FAMILY_STATUS': {49386: 'Married'},
            'NAME_HOUSING_TYPE': {49386: 'House / apartment'},
            'NAME_INCOME_TYPE': {49386: 'Working'},
            'CODE_GENDER': {49386: 'F'}
        }
        self.assertEqual(computed_result, expected_result)
        print(f"Le resultat général pour {existent_id_client} est correct")
 
if __name__ == '__main__':
    unittest.main()

