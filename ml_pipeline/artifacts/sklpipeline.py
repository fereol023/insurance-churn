from dstoolbox.pipeline import DataFrameFeatureUnion
from sklearn.pipeline import Pipeline

from .encoders import *


class SKLPipeline:
    """
    Pipeline SKlearn de machine learning.
    Il encapsule la preproc(encoders) + le modèle.
    Utiliser pour entrainer avec la CV le meilleur modèle + sauvegarder
    """
    def __init__(self):
        self.model = None
        lot1: list[str] = [
            'household_size',
            'age_of_respondent'
        ]

        lot2: list[str] = [
            'country',
            'relationship_with_head',
            'marital_status',
            'education_level',
            'job_type'
        ]

        lot2_bis = ['education_level']

        lot3: list[str] = [
            'location_type',
            'cellphone_access',
            'gender_of_respondent'
        ]

        lot4: list[str] = [
            'year'
        ]

        self.feature_pipeline = DataFrameFeatureUnion(
            [
                ('StandardEncoding', OwnStandardEncoder(columns=lot1)),
                ('LabelEncoding', OwnLabelEncoder(columns=lot2_bis)),
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
