import joblib
import logging
import pandas as pd
from api.ClientDataModel import ClientDataM
from ml_pipeline.artifacts.sklpipeline import SKLPipeline

   
class ChurnC:

    CHURN_MODEL_PATH = 'ml_pipeline/models_registry/mlserver/'

    def __init__(self) -> None:
        self.model = self.load_model(ChurnC.CHURN_MODEL_PATH + '1720038122.joblib')
        self.model_pipeline = SKLPipeline().build_pipeline(model=self.model)
        self.pprocess_pipeline = SKLPipeline().build_pipeline(model=None)

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
        } # order des X > peut import si on le passe par le pipeline de feature eng

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

        # transform into pad dataframe 
        new_data = pd.DataFrame(new_data, index=[0])

        new_data = self.pprocess_pipeline.fit_transform(new_data)
        print(new_data)
        
        # passer au pipeline
        return None
        # 


