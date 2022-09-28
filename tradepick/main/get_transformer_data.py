from re import I
import pandas as pd

from .preprocess import config, get_timestamps, collect_data, plot_closing, plot_gain, compare_stocks
from .models import TorchRNN, rnn_params, transf_params, TransformerModel
from .dataset import GetDataset
from .train import Classifier, plot_predictions, get_predicted_value
from .Analysis import *
from .indicators import simple_moving_average
from .sentiment import get_sentiment
import concurrent.futures


def initialize_transformer(dataset_location, ticker) -> float:
    download_dataset(ticker)
    return run(dataset_location, 'transformer', ticker, True)


# def download_dataset():
#     for idx, stock in enumerate(config.stock_names):

#         timestamps = get_timestamps(config.yrs, config.mths, config.dys)
#         df = collect_data(timestamps, stock, simple_moving_average, True)

#         company_name = df["Company stock name"][0]

#         df.to_csv('Data\{company_ticker}.csv'.format(
#             company_ticker=company_name), index=True)


def download_dataset(ticker):

    timestamps = get_timestamps(config.yrs, config.mths, config.dys)
    df = collect_data(timestamps, ticker, simple_moving_average, True)
    df = df[:-1]
    df = df.tail(200)
    company_name = df["Company stock name"][0]

    df.to_csv('{company_ticker}.csv'.format(
        company_ticker=company_name), index=True)


def run(stock: str, model_type: str, company_name: str, stationary=True) -> float:
    df = get_data(stock)
    df["Company stock name"] = company_name
    dataset = GetDataset(df)
    dataset.get_dataset(scale=False, stationary=stationary)

    train_data, test_data, train_data_len = dataset.split(
        train_split_ratio=0.8, time_period=30)

    train_data, test_data = dataset.get_torchdata()

    x_train, y_train = train_data

    x_test, y_test = test_data

    if model_type == 'lstm':
        params = rnn_params
        model = TorchRNN(rnn_type=params.rnn_type, input_dim=params.input_dim,
                         hidden_dim=params.hidden_dim, output_dim=params.output_dim,
                         num_layers=params.num_layers)
    elif model_type == 'transformer':
        params = transf_params
        model = TransformerModel(params)
    else:
        raise ValueError(
            'Wrong model type selection, select either "1" or "2".')

    clf = Classifier(model)

    clf.train([x_train, y_train], params=params)
    y_scaler = dataset.y_scaler
    # predictions = model(x_test).detach().numpy()

    predictions = clf.predict([x_test, y_test], y_scaler, data_scaled=False)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = df.index[-len(x_test):]
    predictions['Actual'] = y_test[:-1]

    # valid = df[train_data_len:][:-2]
    # valid['Predictions'] = predictions
    # x_val = [str(valid.index[i]).split()[0] for i in range(len(valid))]

    # print(x_val)

    predictions.rename(columns={0: 'Predictions'}, inplace=True)
    if stationary:
        predictions = inverse_stationary_data(old_df=df, new_df=predictions,
                                              orig_feature='Actual', new_feature='Predictions',
                                              diff=12, do_orig=False)
    plot_predictions(df, train_data_len,
                     predictions["Predictions"].values, model_type)

    predicted_value = get_predicted_value()
    # print(predicted_value)
    return float(predicted_value)+1


# if __name__ == '__main__':

#     with open('Data\shared.csv', 'r') as fin:
#         ticker = fin.read()

#     dataset_location = 'Data/'+ticker+'.csv'

#     print(initialize_transformer(dataset_location, ticker))
