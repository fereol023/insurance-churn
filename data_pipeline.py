import pandas as pd
import pyarrow
import uuid
from datetime import datetime
import json
import os

from insurance_churn.ml_pipeline.artifacts.pipeline_loggers import data_logger

#@data_logger

fnames = {
    "claims": "data_files/customer_claims_complaints.xlsx",
    "details": "data_files/policydetails.xlsx",
    "premium": "data_files/policypremium.xlsx"
}

log={
    "dataset_id": "",
    "timestamp": str(datetime.today()), 
    "nb_rows": "",
    "status": "",
    "details": ""
    }

try : 
    claims = pd.read_excel(fnames["claims"])
    details = pd.read_excel(fnames["details"])
    premium = pd.read_excel(fnames["premium"])

    # ajouter le prix au details du contrat
    # left join details x premium on customer id
    # on a tous les details et on veut les prix 
    # si c'est complet, normalement inner join passe aussi
    details_and_premium = pd.merge(left=details, right=premium, how='left', on='CustomerID')

    # les infos supplémenentaires du customer à partir des claims
    # Tt le monde n'a pas forcément fait de claim
    # donc on garde details_n_premium et on lui joint par la gauche 
    # les claims
    full_data = pd.merge(left=details_and_premium, right=claims, how='left', on='CustomerID')
    
    full_data['InceptionDate'] = full_data['InceptionDate'].astype('str')
    full_data = full_data.sort_values(by='InceptionDate')
    train = full_data.iloc[:-400,] 
    test = full_data.iloc[-400:,]

    id = str(uuid.uuid4())
    dfilename = f'data_pipeline_outputs/{id}.parquet.gzip'
    dfilename = f'data_pipeline_outputs/{id}.csv'

    op_root = f'data_pipeline_outputs/{id}'
    train_filename = f'{op_root}/train.csv'
    test_filename = f'{op_root}/test.csv'
    #full_data.to_parquet(f'data_pipeline_outputs/{id}.parquet', engine='pyarrow')
    #full_data.to_parquet(dfilename, compression='gzip')

    if not os.path.exists(op_root):
        os.makedirs(op_root)

    train.to_csv(train_filename)
    test.to_csv(test_filename)

    log['status'] = "success"
    log['dataset_id'] = id
    log['nb_rows'] = len(full_data)
    
except Exception as e:
    log['status'] = "error"
    log['details'] = str(e)
    raise e

finally:
    with open("data_pipeline_outputs/logs/logs.json", "a") as f:
        json.dump(log, f)