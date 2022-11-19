from django.shortcuts import render
from django.views import View
from .scrape import get_tweet
from .forms import SearchForm


"""class Index(View):
    template = 'index.html'

    def get(self, request):
        form = SearchForm()
        success_url = '/result/'
        return render(request, self.template, {'form':form})"""

def get_username(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SearchForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = SearchForm()

    return render(request, 'index.html', {'form': form})

def InputWebView(requests):
    form = WebInputForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            request.session['web_input'] = request.POST['web_input']
            return redirect('add_web')