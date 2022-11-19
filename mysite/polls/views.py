from django.shortcuts import render
from django.views import View
from .scrape import get_tweet
from .forms import SearchForm
from django.http import HttpResponseRedirect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import yake

import requests
from bs4 import BeautifulSoup

from pathlib import Path

HERE = Path(__file__).parent



"""class Index(View):
    template = 'index.html'

    def get(self, request):
        form = SearchForm()
        success_url = '/result/'
        return render(request, self.template, {'form':form})"""

def get_username(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)

        print(form.errors)

        if form.is_valid():
            username = form.cleaned_data.get("username")
            print("Cleaned username: ", username)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/result/' + username)

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SearchForm()

    return render(request, 'index.html', {'form': form})

def result(request,username):
    name = username

    tweets = get_tweet(username,100)

    return render(request, 'result.html', {'tweets': tweets})

def InputWebView(requests):
    form = WebInputForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            request.session['web_input'] = request.POST['web_input']
            return redirect('add_web')

        return render(request, self.template)

def predictSentiment(text):
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'

    tokenizer = pickle.load(open(HERE / 't.sav', 'rb'))
    model = tf.keras.models.load_model(HERE / 'SA.h5')

    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    result = model.predict(padded)[0][0]
    if result >= 0.5:
        return "positive"
    else:
        return "negative"

# def keyextract(text):
#     kw_extractor = yake.KeywordExtractor()
#     language = "en"
#     max_ngram_size = 3
#     deduplication_threshold = 0.9
#     numOfKeywords = 10
#     custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#     keywords = custom_kw_extractor.extract_keywords(text)
#     keywords_set = set()
#     for kw in keywords:
#         keywords_set.add(kw[0])
#     return keywords_set
#
# def allStocks(words):
#     stock
#     for word in words:
#         url = 'https://www.investing.com/equities/' + word.lower():
#         page = requests.get(url)
#         if  page.status_code == 200:
#             soup = BeautifulSoup(page.text, 'html.parser')
#             company = soup.find('h1', {'class': 'text-2xl font-semibold instrument-header_title__GTWDv mobile:mb-2'}).text
#             price = soup.find('div', {'class': 'instrument-price_instrument-price__3uw25 flex items-end flex-wrap font-bold'}).find_all('span')[0].text
#             change = soup.find('div', {'class': 'instrument-price_instrument-price__3uw25 flex items-end flex-wrap font-bold'}).find_all('span')[2].text
#             return [company, price, change]

def models(request):

    sentence = 'default sentence'

    if request.GET.get('sentence'):
        sentence = request.GET.get('sentence')

    result = predictSentiment(sentence)

    return render(request, 'result.html', {'result':result})
