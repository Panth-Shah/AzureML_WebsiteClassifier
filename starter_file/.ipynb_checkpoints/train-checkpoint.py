from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df = df.copy()
    
    # Drop URL column
    df = df.drop('URL', axis=1)
    
    # Extract datetime features
    for column in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    
    df['REG_YEAR'] = df['WHOIS_REGDATE'].apply(lambda x: x.year)
    df['REG_MONTH'] = df['WHOIS_REGDATE'].apply(lambda x: x.month)
    df['REG_DAY'] = df['WHOIS_REGDATE'].apply(lambda x: x.day)
    df['REG_HOUR'] = df['WHOIS_REGDATE'].apply(lambda x: x.hour)
    df['REG_MINUTE'] = df['WHOIS_REGDATE'].apply(lambda x: x.minute)
    
    df['UPD_YEAR'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: x.year)
    df['UPD_MONTH'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: x.month)
    df['UPD_DAY'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: x.day)
    df['UPD_HOUR'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: x.hour)
    df['UPD_MINUTE'] = df['WHOIS_UPDATED_DATE'].apply(lambda x: x.minute)
    
    df = df.drop(['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis=1)
    
    
    # One-hot encode categorical features
    for column in ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']:
        df[column] = df[column].apply(lambda x: x.lower() if str(x) != 'nan' else x)
    
    df = onehot_encode(
        df,
        column_dict={
            'CHARSET': 'CH',
            'SERVER': 'SV',
            'WHOIS_COUNTRY': 'WC',
            'WHOIS_STATEPRO': 'WS'
        }
    )
    
    # Fill missing values
    missing_value_columns = df.columns[df.isna().sum() > 0]
    
    for column in missing_value_columns:
        df[column] = df[column].fillna(df[column].mean())
    
    # Split df into X and y
    y = df['Type'].copy()
    X = df.drop('Type', axis=1).copy()

    
    return X, y


def main():
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/Panth-Shah/nd00333-capstone/master/Dataset/malicious_website_dataset.csv', help='Dataset URL')
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Get the dataset
    dataset = Dataset.Tabular.from_delimited_files(args.data_url)
    
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
        
    # Perform Data pre-processing
    ds_data = dataset.to_pandas_dataframe()
    
    X, y = preprocess_inputs(ds_data)
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    # Scale X with a standard scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Remove feature columns that were reduced to single-value columns during the train-test split
    single_value_columns = X_train.columns[[len(X_train[column].unique()) == 1 for column in X_train.columns]]
    
    X_train = X_train.drop(single_value_columns, axis=1)
    X_test = X_test.drop(single_value_columns, axis=1)
    

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/capstone_model.joblib')

if __name__ == '__main__':
    main()
