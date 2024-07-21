import joblib
import logging
import pandas as pd
from api.ClientDataModel import ClientDataM
from ml_pipeline.artifacts.sklpipeline import SKLPipeline

   
class ChurnC:

    CHURN_MODEL_PATH = 'ml_pipeline/models_registry/mlserver/'

    def __init__(self) -> None:
        self.model = self.load_model(ChurnC.CHURN_MODEL_PATH + '1721516304.joblib')
        self.model_pipeline = SKLPipeline().build_pipeline(model=self.model)
        self.pprocess_pipeline = SKLPipeline().build_pipeline(model=None)
        self.output = None
        logging.info("Model and pipeline loaded correctly !")

    def load_model(self, path):
        """charger un modele"""
        # assert path exists
        try:
            return joblib.load(path)
        except Exception as e:
            print(f'Error in model loading : {e}')

    def get_features_names(self, pipeline):
        """TODO"""
        # if hasattr(model, 'feature_names_in'):
        #     return model.feature_names_in_
        # elif hasattr(model, 'feature_importances_'):
        #     return model.feature_names_
        # else:
        #     raise AttributeError('The model does not have features names attributes')
        return pipeline['Model'].feature_names_in_


    def debug(self):
        """TODO"""
        new_data = {k:"" for k in self.get_features_names(self.model_pipeline)}
        return new_data

    def predict(self, client: ClientDataM):
        """TODO"""
        
        # le model def avec le bon ordre des X
        new_data_model = {k:0 for k in self.get_features_names(self.model_pipeline)}

        # un dict temp pour recup la data et la passer au preproc
        new_data = {
             "policy_duration_v1": "",
             "inception_date_year_v1": "",
             "inception_date_month": "",
             "package": "",
             "has_additionnal_policies": "",
             "discount_v1": "",
             "premium": "",
             "age_v1": "",
             "gender": "",
             "total_claims_value_v1": "",
             "total_claims_number": "",
             "number_of_complaints": ""
        } 

        # recup la data de l'api
        new_data["policy_duration_v1"] = client.policy_duration_v1
        new_data["inception_date_year_v1"] = client.inception_date_year_v1
        new_data["inception_date_month"] = client.inception_date_month
        new_data["package"] = client.package
        new_data["has_additionnal_policies"] = client.has_additionnal_policies
        new_data["discount_v1"] = client.discount_v1
        new_data["premium"] = client.premium
        new_data["age_v1"] = client.age_v1
        new_data["gender"] = client.gender
        new_data["total_claims_value_v1"] = client.total_claims_value_v1
        new_data["total_claims_number"] = client.total_claims_number
        new_data["number_of_complaints"] = client.number_of_complaints

        # transform into pd dataframe 
        new_data = pd.DataFrame(new_data, index=[0])
        # exec pipeline de preproc
        print(' --- start preproc ! ')
        new_data = self.pprocess_pipeline.fit_transform(new_data)
        print(' --- out of prproc ! ')
        # assigner les data preproc aux model final (pour assurer le bon ordre)
        for k,v in new_data.items():
            new_data_model[k] = v
        # ensuite convertir en pd dataframe
        new_data_model_df = pd.DataFrame(new_data_model, index=[0])
        print(' --- ready for predict ! ')
        # print(new_data_model_df.T)
        required_features = self.model.feature_names_in_
        missings_features = set(required_features) - set(new_data_model_df.columns)
        assert missings_features == set(), f"Missing features : {missings_features}"
        
        # passer au model
        output = self.model.predict(new_data_model_df)
        
        self.output = output[0]
        # print(f'OUTPUT : {self.output}')
        return self
        
    def get_output(self):
        target_dict = {'0': 'churn_no', '1': 'churn_yes'}
        return target_dict.get(str(self.output))
