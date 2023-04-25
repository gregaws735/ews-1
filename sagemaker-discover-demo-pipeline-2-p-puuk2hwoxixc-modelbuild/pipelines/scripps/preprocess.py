import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
from time import gmtime, strftime                 # For labeling SageMaker models, endpoints, etc.
import sys                                        # For writing outputs to notebook
import math                                       # For ceiling function
import json                                       # For parsing hosting outputs
import os                                         # For manipulating filepath names
import sagemaker 
import pathlib
import logging
import argparse

 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role


from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

base_dir = "/opt/ml/processing"

if __name__ == "__main__":
    
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_group_name", type=str, required=True)
    args = parser.parse_args()
    feature_group_name = args.feature_group_name
    logger.debug(feature_group_name)
    feature_group_name = "Scripps-Feature-Store-Full"
    bucket='sagemaker-us-west-2-326229093341' #sagemaker.Session().default_bucket()
    prefix = 'sagemaker/FS-xgboost-dm'
    #role = get_execution_role()
    region = 'us-west-2'#boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
    featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )
    
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)

    fs_query = feature_group.athena_query()
    fs_table = fs_query.table_name
    query_string = 'SELECT * FROM "'+fs_table+'"'
    print('Running ' + query_string)

    fs_query.run(query_string=query_string, output_location='s3://'+bucket+'/'+prefix+'/fs_query_results/')
    fs_query.wait()
    df = fs_query.as_dataframe()

    df = df.drop(['fs_id', 'fs_time', 'write_time', 'api_invocation_time', 'is_deleted'], axis=1)
    y = df.pop("cs_impression")
    y_pre = y.to_numpy().reshape(len(y), 1)
    
    """
    columns_to_drop = ["WO_DISPLAY_DAYS","WO_PROGRAM_NAME","WO_GENRE","WO_INVENTORY_CODE_NAME","WO_PROGRAM_INT","WO_DAYS","WO_LABEL_DERIVED","WO_DAYPARTS_LABEL_TIME_PERIOD","CS_DAY_OF_WEEK","DURATION_HH_MM","CS_DEMOGRAPHICS","CS_DATA_STREAM","FS_ID","FS_time"]

    #cols = [c for c in df.columns if not c.startswith(col_prefix)]
    cols = [c for c in df.columns if c in columns_to_drop ]
    for col in cols:
        df.drop(col, axis=1, inplace=True)
    """ 

    X_pre = df
    X = np.concatenate((y_pre, X_pre), axis=1)
    #np.random.shuffle(X)

    # Split in Train, Test and Validation Datasets
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    train_rows = np.shape(train)[0]
    validation_rows = np.shape(validation)[0]
    test_rows = np.shape(test)[0]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    validation = pd.DataFrame(validation)
    # Convert the label column to integer
    train[0] = train[0].astype(int)
    test[0] = test[0].astype(int)
    validation[0] = validation[0].astype(int)
    # Save the Dataframes as csv files
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)

    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)