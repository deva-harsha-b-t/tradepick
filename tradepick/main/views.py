from urllib import response
from django.shortcuts import render
from django.http import HttpResponse




def index(response):
    StockData = {
        'name': 'Apple',
        'price': '$1212',
        'sentiment':'Negative'

    }
    return render(response, "main/base.html", StockData)

def processstock(request):
    name = request.GET['stockname']
    stockData = {
        'name': 'select stock',
        'price': '$0',
        'sentiment':'no value'

    }
    if name == 'AAPL':
        stockData = {
        'name': 'Apple',
        'price': '$12313',
        'sentiment':'Positive'
        }
    elif name == 'DIS':
        stockData = {
            'name': 'Disney',
            'price': '$575',
            'sentiment':'Negative'
        }
    elif name == 'NKE':
        stockData = {
            'name': 'Nike',
            'price': '$3467',
            'sentiment':'Positive'
        }
        
    return render(request,"main/base.html", stockData)