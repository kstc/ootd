from django import forms

class UploadForm(forms.Form):
	picture = forms.FileField()
	women_choices = (
		('1', 'accessories'),
		('2', 'coats/sweater'),
		('3', 'dresses'),
		('4', 'pants'),
		('5', 'shorts'),
		('6', 'skirts'),
		('7', 'tops')
	)
	men_choices = (
		('1', 'accessories'),
		('2', 'pants'),
		('3', 'shirts'),
		('4', 'shorts'),
		('5', 'suits')
	)
	
	sex = forms.ChoiceField(widget=forms.RadioSelect, choices=(('1', 'Men'), ('2', 'women')))
	clothe_type = forms.ChoiceField(choices=(women_choices + men_choices))
