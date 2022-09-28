from urllib import response
from django.shortcuts import render
from django.http import HttpResponse
from .sentiment_transformer import *


def index(response):
    StockData = {
        'name': 'Apple',
        'price': '$1212',
        'sentiment': 'Negative'

    }
    return render(response, "main/base.html", StockData)


def ignite(ticker):
    sentiment_val, intrensic_val = gear_up(ticker)
    return sentiment_val, intrensic_val


def processstock(request):
    name = request.GET['tickerforsearch']
    print(name)
    name = str(name)
    sentiment_val, intrensic_val = ignite(name)
    #sentiment_val, intrensic_val = 1, 131
    stockData = {
        'name': 'select stock',
        'price': '$0',
        'sentiment': 'no value'

    }
    # if name == 'AAPL':
    #     stockData = {
    #         'name': 'Apple',
    #         'price': '$12313',
    #         'sentiment': 'Positive'
    #     }
    # elif name == 'DIS':
    #     stockData = {
    #         'name': 'Disney',
    #         'price': '$575',
    #         'sentiment': 'Negative'
    #     }
    # elif name == 'NKE':
    #     stockData = {
    #         'name': 'Nike',
    #         'price': '$3467',
    #         'sentiment': 'Positive'
    #     }
    sentiment = 'Positive'
    if sentiment_val < 0:
        sentiment = 'Negative'

    p_price = '$' + str(intrensic_val)

    stockData = {
        'name': name,
        'price': p_price,
        'sentiment': sentiment,
    }

    return render(request, "main/base.html", stockData)
