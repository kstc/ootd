from django import forms

class UploadForm(forms.Form):
	picture = forms.FileField(label="Choose an image")
	women_choices = (
		('1', 'accessories'),
		('2', 'coats/sweaters'),
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

	sex = forms.ChoiceField(widget=forms.RadioSelect, choices=(('1', 'Men\'s'), ('2', 'Women\'s')), label="What type of clothes do you want to see?")
	sex.initial = '1'
	men_clothing_type = forms.ChoiceField(choices=(men_choices), label="")
	women_clothing_type = forms.ChoiceField(choices=(women_choices), label="")
