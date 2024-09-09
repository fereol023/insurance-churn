from dstoolbox.pipeline import DataFrameFeatureUnion
from sklearn.pipeline import Pipeline

#from .encoders import *
from ml_pipeline.artifacts.encoders import * #(mode prod)


class SKLPipeline:
    """
    Pipeline SKlearn de machine learning.
    Il encapsule la preproc(encoders) + le modèle.
    Utiliser pour entrainer avec la CV le meilleur modèle + sauvegarder
    """
    def __init__(self):
        self.model = None
        target: str = 'churn' # package # premium

        lot1: list[str] = [ 
            "policy_duration_v1",
            "premium",
            "age_v1",
            "total_claims_value_v1"
        ]

        lot2: list[str] = [
             "package"
        ]


        lot3: list[str] = [
            "gender",
            "discount_v1",
            "has_additionnal_policies",

        ]

        lot4: list[str] = [
            "inception_date_year_v1",
            "inception_date_month",
            "total_claims_number",
            "number_of_complaints"
        ]

        self.feature_pipeline = DataFrameFeatureUnion(
            [
                #('UnbalancedChecking', Balancer(target=[target]))
                ('StandardEncoding', OwnStandardEncoder(columns=lot1)),
                ('LabelEncoding', OwnLabelEncoder(columns=lot2)),
                ('OneHotEncoding', OwnOneHotEncoder(columns=lot3)),
                ('NotEncoding', DataFeatureExtractor(columns=lot4))
            ]
        )

    def build_pipeline(self, model=None):
        """
        Pour construire le pipeline preproc+model.
        :param model: sklearn model, garder None pour pipeline de feature eng.
        """
        try:
            if model is None:
                return Pipeline([
                    ('FeatureEngineering', self.feature_pipeline)
                ])
            else:
                self.model = model
                return Pipeline([
                    ('FeatureEngineering', self.feature_pipeline),
                    ('Model', self.model)
                ])
        except Exception as e:
            print("Exception occurs in SKLPipeline.build_pipeline()")
