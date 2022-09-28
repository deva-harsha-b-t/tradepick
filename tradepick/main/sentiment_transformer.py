from .sentiment import *
from .get_transformer_data import *

# df = get_sentiment("TSLA", 100)
# # print(get_top_headline())
# print("Sentiment for TSLA is : ", df)

# with open('Data\shared.csv', 'r') as fin:
#     ticker = fin.read()

# dataset_location = 'Data/'+ticker+'.csv'

# print(initialize_transformer(dataset_location, ticker))


def gear_up(ticker):
    sentiment_val = get_sentiment(ticker, 100)
    dataset_location = ticker+'.csv'
    intrensic_val = initialize_transformer(dataset_location, ticker)

    return sentiment_val, intrensic_val
