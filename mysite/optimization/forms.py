# optimization/forms.py
from django import forms

class OptimizationForm(forms.Form):
    objective_function = forms.CharField(label='Objective Function', max_length=200)
    constraints = forms.CharField(widget=forms.Textarea, label='Constraints')
    method = forms.ChoiceField(choices=[('simplex', 'Simplex Method'), ('graphical', 'Graphical Method')], label='Method')
