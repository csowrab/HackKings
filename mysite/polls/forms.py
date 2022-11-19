from django import forms

class SearchForm(forms.Form):
    username = forms.CharField()

    def clean_title():
        cleaned_data = self.cleaned_data
        username = cleaned_data.get('username')

        return username

    """def clean(self):
        super(forms.Forms, self).clean()

        username = self.cleaned_data.get('username')

        if username == "":
            raise forms.ValidationError(['This field is required.'])"""