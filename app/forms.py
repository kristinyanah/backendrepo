
from flask_wtf import Form
from wtforms import StringField , FileField ,IntegerField , HiddenField , PasswordField , BooleanField , SubmitField , TextAreaField , RadioField , SelectField
from wtforms.fields.html5 import DateField , TelField
from wtforms.validators import DataRequired , Email , EqualTo , ValidationError , Length , Required 
from flask_wtf.file import FileRequired


class SelectForm(Form):

    substrate = TextAreaField(label="Substrate SMILES" , validators=[DataRequired()])
    smiles = TextAreaField(label="Enzyme SMILES" , validators=[DataRequired()])

    interaction = SelectField(label= "Interaction" , choices=[('0','0'),
                                                        ('1','1')  ],
                                            validators=[DataRequired()])

    
    submit = SubmitField(label="Search" )

