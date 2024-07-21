import shap
import warnings
import matplotlib.pyplot as plt
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from commons import variables
from artifacts import *
from artifacts.sklpipeline import SKLPipeline
warnings.filterwarnings('ignore')


@line
def main(csv_data_path, version='v1_0'):
    """
    Entrainer le modèle. # ps changer à la mano (rf par defaut)
    :param csv_data_path: chemin vers le df 
    """
    data = pd.read_csv(csv_data_path)

    target = variables[version]['target_name']
    x_features_names = variables[version]['explained_by']
    x_features = data[x_features_names]
    y_target = data[target]

    param_rf = {
        'n_estimators': [int(x) for x in np.linspace(10, 80, 10)],  # nombre d'arbres dans la forÃªt 10 20 30 ... 80
        'max_depth': [2, 1000],  # nombre max de niveaux dans un arbre
        'min_samples_split': [5, 10],
        # nombre min d'echantillons (bootstrap) necessaire au niveau d'un noeud pour juger de le spliter
        'min_samples_leaf': [1, 200],  # nombre min de samples requis Ã  chaque node
        'max_features': ['log2', 'sqrt']  # nombre de features Ã  considÃ©rer auto = et sqrt =
    }

    rf_cv = GridSearchCV(RandomForestClassifier(), param_grid=param_rf, cv=4, n_jobs=1)

    ml_pipeline = SKLPipeline().build_pipeline(model=rf_cv)
    
    #rf_cv_inc_pipeline = GridSearchCV(RandomForestClassifier(), param_grid=param_rf, cv=4, n_jobs=1, error_score='raise')
    #rf_cv_inc_pipeline.fit(x_features, y_target)

    ml_pipeline.fit(x_features, y_target)
    dump(rf_cv.best_estimator_, str(MyPaths.fittedModel))

    # feature importance dans la phase d'apprentissage
    feats = {}
    for colname, importance in list(zip(x_features_names, rf_cv.best_estimator_.feature_importances_)):
        feats[colname] = importance

    mdi_importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    mdi_importances.sort_values(by='Gini-importance').plot(kind='bar', rot=30, figsize=(15, 8))
    plt.savefig(f'ml_pipeline/models_registry/rf_features_importance_{round(time.time())}.png')
    plt.close()

    print(f'Model saved with params : {rf_cv.best_params_}')


if __name__ == '__main__':
    train_data_path = "data_processed/period0/train.csv"
    main(train_data_path)
