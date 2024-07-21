import shap
import warnings
import matplotlib.pyplot as plt
from joblib import load, dump

from commons import variables
from artifacts import *
from artifacts.sklpipeline import SKLPipeline
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

@line
def main(csv_data_path, version='v1_0'):
    
    data = pd.read_csv(csv_data_path)

    target = variables[version]['target_name']
    x_features_names = variables[version]['explained_by']
    X_test = data[x_features_names]
    y_test = data[target]

    backup_model = load('ml_pipeline/models_registry/mlserver/1721516304.joblib')
    print(f'Model loaded :  {backup_model} :: {X_test.shape}')
    pprocess = SKLPipeline().build_pipeline(model=None)
    X_test_transf = pprocess.fit_transform(X_test)

    required_features = backup_model.feature_names_in_
    missings_features = set(required_features) - set(X_test_transf.columns)
    assert missings_features == set(), f"Missing features : {missings_features}"

    # test
    y_pred = backup_model.predict(X_test_transf)

    # evaluate
    res, rule = score_and_validate_results(y_test, y_pred)
    print(f"Perfs:\n {res}")

    if rule:
        # save
        X_test_transf['churn_pred'] = y_pred
        X_test_transf['churn_obs'] = y_test
        X_test['churn_pred'] = y_pred
        X_test['churn_obs'] = y_test

        outputs_rootpath = 'ml_pipline_data_outputs/'
        if not os.path.exists(outputs_rootpath):
            os.makedirs(outputs_rootpath)
        X_test.to_csv(outputs_rootpath+'test_res_raw.csv')
        X_test.to_csv(outputs_rootpath+'test_res_transf.csv')


def score_and_validate_results(y_true, y_pred):

    def val_rule_1(s_d):
        # le plus grave dans le cas de la resil 
        # serait de predire un fn
        # on va min les fn (moins de 5%)
        # les fp peuvent tjrs avoir des comm marketing etc.. pas de soucis
        # mais les fn pourraient echapper à la stratégie et fausser les estimations
        return s_d['fn'] <= 0.05
    
    def val_rule_2(s_d):
        return True
    
    cm = confusion_matrix(y_true, y_pred, normalize='all') # 'pred' ou 'true' # au denom les vraies valeurs ou les pred etc..
    d = {
        'tn': cm.ravel()[0],
        'fp': cm.ravel()[1],
        'fn': cm.ravel()[2],
        'tp': cm.ravel()[3]
    }
    
    return d, val_rule_1(d)

if __name__=='__main__':
    import os
    print(os.getcwd())
    test_data_path = "data_processed/period0/test.csv"
    main(test_data_path)
