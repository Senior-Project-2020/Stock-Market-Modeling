import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def window_data(data, window_size):
    '''
    Windows data into back to back intervals that include the previous N samples

    Args:
        data (pd.DataFrame): Dataframe object of saved stock data to be windowed
        window_size (int): Size of cooresponding windows to create

    Returns:
        np.array: numpy array of windowed data in shape (N, window_size*5)
    '''
    windowed_data = list()
    next_close = list()

    for i in np.arange(0, len(data)-window_size):
        windowed_data.append(data.iloc[i:i+window_size, :5].values.flatten())
        next_close.append(data.iloc[i+window_size, 5])

    return np.array(windowed_data), np.array(next_close)


def load_windowed_dataset(companies, window_size=5, inflation_adjusted=True, train_size=0.8):
    '''
    Loads the data of companies from local directories

    Args:
        companies (List): list of company stock symbols to load from csv
        window_size (int): number to specify previous n results to include in windowed data
        inflation_adjusted (bool): this is a flag to load the saved infaltion adjusted data or not
        train_size (float 0 - 1): percentage of data to be used as the train set

    Returns:
        np.array: Values of stock data in coresponding windows in shape (N, window_size*5)
    '''
    X_train = np.ndarray(shape=(0, 25))
    X_test = np.ndarray(shape=(0, 25))
    y_train = np.array([])
    y_test = np.array([])

    data_dir = 'Inflation_Adjusted' if inflation_adjusted else 'Yahoo_Data'

    for company in companies:
        try:
            company_data = pd.read_csv(f'{data_dir}/{company}.csv', index_col=0)
            company_data = remove_null_rows(company_data)
        except Exception:
            print(f'{company} not found')
            continue

        print(f'Loaded: {company}')

        X, y = window_data(company_data.drop(columns=['Date', 'Adj Close']), 5)

        index = math.floor(len(company_data)*train_size)
        X_train = np.concatenate((X_train, X[:index]))
        X_test = np.concatenate((X_test, X[index:]))
        y_train = np.concatenate((y_train, y[:index]))
        y_test = np.concatenate((y_test, y[index:]))
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(X_train.shape)
    print(y_train.shape)
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X_train, X_test)))

    return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test


def remove_null_rows(data):
    '''
    Remove null rows from data

    Args:
        data (pd.DataFrame): data to remove null rows from
    
    Returns:
        pd.DataFrame: the same dataframe without null rows removed
    '''
    null_columns=data.columns[data.isnull().any()]
    null_rows = data[data.isnull().any(axis=1)][null_columns].index
    return data.drop(null_rows).reset_index(drop=True)


def build_cnn_model():
    cnn_model = Sequential()
    cnn_model.add(layers.Reshape((5, 5), input_shape=(25,)))

    cnn_model.add(layers.Conv1D(filters=10, kernel_size=3))
    cnn_model.add(layers.BatchNormalization(scale=True))
    cnn_model.add(layers.Activation("relu"))

    cnn_model.add(layers.GlobalAveragePooling1D())

    cnn_model.add(layers.Dense(25))
    cnn_model.add(layers.Activation("relu"))

    cnn_model.add(layers.Dropout(0.3))

    cnn_model.add(layers.Dense(50))
    cnn_model.add(layers.Activation("relu"))

    cnn_model.add(layers.Dropout(0.3))

    cnn_model.add(layers.Dense(100))
    cnn_model.add(layers.Activation("relu"))

    cnn_model.add(layers.Dropout(0.3))

    cnn_model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

    cnn_model.summary()
    cnn_model.compile(optimizer='rmsprop', loss='mse')
    return cnn_model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    print("\nTrain Data")
    print(f'R2 score: {r2_score(y_train, train_prediction)}')
    print(f'Mean Squared Error: {mean_squared_error(y_train, train_prediction)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_train, train_prediction)}')

    print("\nTest Data")
    print(f'R2 score: {r2_score(y_test, test_prediction)}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, test_prediction)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, test_prediction)}')


def main():
    companies = pd.read_json("s-and-p-500-companies/data/constituents_json.json")

    X_train, y_train, X_test, y_test = load_windowed_dataset(list(companies.loc[:, 'Symbol']))

    print(X_train.shape)
    print(y_train.shape)
    model = build_cnn_model()
    model.fit(X_train, y_train, epochs=50)

    evaluate_model(model, X_train, X_test, y_train, y_test)

    pass


if __name__ == '__main__':
    main()