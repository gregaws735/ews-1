import pandas as pd
import glob
import os
import numpy as np
import argparse
import logging

import boto3
import os
import pathlib
from concurrent import futures

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    input_data = args.input_data
    #input_data = 's3://sagemaker-us-west-2-326229093341/scripps/scripps-2023-04-16T23-00-09/'

    base_dir = "/opt/ml/processing"
    #base_dir = "/root/scripps/processing"
    data_dir = f"{base_dir}/data"

    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    bucket_name = input_data.split("/")[2]
    s3_prefix = "/".join(input_data.split("/")[3:])

    print(bucket_name)
    print(s3_prefix)

    s3_client = boto3.client('s3')
    #bucket_name = 'sagemaker-us-west-2-326229093341'
    #s3_prefix = 'scripps/scripps-2023-04-16T23-00-09/'

    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=s3_prefix)
    s3_object_keys = []
    for content in response.get('Contents', []):
        #print(content['Key'])
        s3_object_keys.append(content['Key'])
    #print(s3_object_keys)    

    #relative_path = './data'
    #bucket_name = 'bucket_name'
    #s3_object_keys = [] # List of S3 object keys
    max_workers = 5

    #abs_path = os.path.abspath(relative_path)
    #s3 = boto3.client('s3')

    def fetch(key):
        file = f'{data_dir}/{key.split("/")[-1]}'
        dir_name = os.path.dirname(file) 
        #os.makedirs(dir_name, exist_ok=True)  
        with open(file, 'wb') as data:
            s3_client.download_fileobj(bucket_name, key, data)
        return file


    def fetch_all(keys):

        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_key = {executor.submit(fetch, key): key for key in keys}

            print("All URLs submitted.")

            for future in futures.as_completed(future_to_key):

                key = future_to_key[future]
                exception = future.exception()

                if not exception:
                    yield key, future.result()
                else:
                    yield key, exception


    for key, result in fetch_all(s3_object_keys):
        logger.debug(f'key: {key}  result: {result}')


    #dirname = os.getcwd()
    #dir_path_name = 'data/scripps/scripps-2023-04-16T23-00-09' 
    #dir_path = os.path.join(dirname, dir_path_name)
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = (pd.read_csv(f, header=0) for f in all_files)
    df   = pd.concat(df_from_each_file, ignore_index=True)    


    col_prefix = "FP"

    columns_to_drop = ["WO_DISPLAY_DAYS","WO_PROGRAM_NAME","WO_GENRE","WO_INVENTORY_CODE_NAME","WO_PROGRAM_INT","WO_DAYS","WO_LABEL_DERIVED","WO_DAYPARTS_LABEL_TIME_PERIOD","CS_DAY_OF_WEEK","DURATION_HH_MM","CS_DEMOGRAPHICS","CS_DATA_STREAM","FS_ID","FS_time"]

    y = df.pop("CS_IMPRESSION")
    y_pre = y.to_numpy().reshape(len(y), 1)

    #cols = [c for c in df.columns if not c.startswith(col_prefix)]
    cols = [c for c in df.columns if c in columns_to_drop ]
    for col in cols:
        df.drop(col, axis=1, inplace=True)


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
