from django.shortcuts import render
from django.views import View
from .scrape import get_tweet
from .forms import SearchForm
from django.http import HttpResponseRedirect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

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



def models(request):

    sentence = 'default sentence'

    if request.GET.get('sentence'):
        sentence = request.GET.get('sentence')

    result = predictSentiment(sentence)

    return render(request, 'result.html', {'result':result})
