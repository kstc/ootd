from django import forms

class UploadForm(forms.Form):
	picture = forms.FileField()
	sex = forms.ChoiceField(widget=forms.RadioSelect, choices=(('1', 'Men'), ('2', 'Women')))
