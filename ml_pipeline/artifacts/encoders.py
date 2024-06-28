import pandas as pd
from artifacts import *
from joblib import load, dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class OwnStandardEncoder(BaseEstimator, TransformerMixin):
    """
    Renvoie un pd df restreint sur les variables quantitatives.
    Ne fait aucun traitement.
    """

    def __init__(self, columns=None):
        super().__init__()
        self.x: pd.DataFrame = None  # pour les données dans la methode fit
        self.columns: list[str] = columns
        # loader le standard encoder fitted sinon un encoder vide
        if os.path.exists(str(MyPaths.estimatorFittedStandardEncoder)):
            self.stdEncoder = load(str(MyPaths.estimatorFittedStandardEncoder))
        else:
            self.stdEncoder = StandardScaler()

    def fit(self, x, y=None):
        if self.columns is not None:
            self.x = x[self.columns]
        else:
            print(f'Warning : Columns to fit are not specified. Encoder will try to fit on every columns of dataset.')
        try:
            self.stdEncoder.fit(self.x)
            return self
        except Exception as e:
            print(f'Exception occurs when fitting_OSE :: {e}')
        finally:
            dump(self.stdEncoder, str(MyPaths.estimatorFittedStandardEncoder))

    def transform(self, x):
        try:
            if self.columns is not None:
                x = x[self.columns]
            return pd.DataFrame(self.stdEncoder.transform(x), columns=self.columns)
        except Exception as e:
            print(f'Exception occurs in_OSE_transform :: {e}')


class OwnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Se base sur le labelEncoder de sklearn.
    Renvoie un pd dataframe au lieu d'un <class 'numpy.ndarray'>.
    /!\ apparemment il y a une option qui permet de renvoyer du pd.df au lieu de np.ndarray dans la màj.
    Sauvegarde l'encodeur pour la reproductibilité.
    """

    def __init__(self, columns=None):
        super().__init__()
        self.x: pd.DataFrame() = None
        self.columns: list[str] = columns
        if os.path.exists(str(MyPaths.estimatorFittedLabelEncoder)):
            self.lblEncoder = load(str(MyPaths.estimatorFittedLabelEncoder))
        else:
            print('On charge un estimateur sklearn vierge')
            self.lblEncoder = LabelEncoder()

    def fit(self, x, y=None):
        if self.columns is not None:
            self.x = x[self.columns]
            #print(self.x.head(1).T)
        else:
            print(f'Warning : Columns to fit are not specified. Encoder will try to fit on every columns of dataset.')
        try:
            self.lblEncoder.fit(self.x)
            #print(self.lblEncoder.get_params())
            #print(self.lblEncoder.get_metadata_routing())
            dump(self.lblEncoder, str(MyPaths.estimatorFittedLabelEncoder))
            return self
        except Exception as e:
            print(f'Exception occurs when fitting_OLE :: {e}')

    def transform(self, x):
        try:
            if self.columns is not None:
                x = x[self.columns]
            return pd.DataFrame(self.lblEncoder.transform(x), columns=self.columns)
        except Exception as e:
            print(f'Exception occurs in_OLE_transform :: {e}')


class OwnOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.x: pd.DataFrame() = None
        self.columns: list[str] = columns
        if os.path.exists(str(MyPaths.estimatorFittedOneHotEncoder)):
            self.ohe = load(str(MyPaths.estimatorFittedOneHotEncoder))
        else:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        try:
            if self.columns is not None:
                X = X[self.columns]
            self.ohe.fit(X)
            # Récupération des noms de colonnes + categories pour l'interprétabilité
            # Ex : "situation_-1", "situation_10", etc..
            self.feature_category_pairs = []
            for i, feature in enumerate(self.ohe.feature_names_in_):
                feature_category = product([feature], self.ohe.categories_[i])
                for pair in feature_category:
                    self.feature_category_pairs.append("{}_{}".format(*pair))
            dump(self.ohe, str(MyPaths.estimatorFittedOneHotEncoder))
            return self
        except Exception as e:
            print(f'Exception occurs in_OHE_fit :: {e}')

    def transform(self, X):
        try:
            if self.columns is not None:
                X = X[self.columns]
            return pd.DataFrame(self.ohe.transform(X), columns=self.feature_category_pairs)
        except Exception as e:
            print(f'Exception occurs in_OHE_transform :: {e}')


class DataFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        super().__init__()
        self.x: pd.DataFrame() = None
        self.columns: list[str] = columns

    def fit(self, x, y=None):
        if self.columns is not None:
            self.x = x[self.columns]
        else:
            print(f'Warning : Columns to fit are not specified. Encoder will try to fit on every columns of dataset.')
        try:
            return self
        except Exception as e:
            print(f'Exception occurs when fitting_DFE :: {e}')

    def transform(self, x):
        try:
            if self.columns is not None:
                x = x[self.columns]
            return pd.DataFrame(self.x, columns=self.columns)
        except Exception as e:
            print(f'Exception occurs in_DFE_transform :: {e}')
