from app import app
from flask import render_template , redirect , url_for , request
from app import prediction
from app.forms import * 

@app.route('/',methods=['GET','POST'])
def index():
    form = SelectForm() 
    if form.validate_on_submit():
        substrate = form.substrate.data.strip()
        smiles = form.smiles.data.strip()
        interaction = form.interaction.data.strip()
        smiles,sequence,interaction_probability,binary_class = prediction.callPredictionMain(substrate,smiles)
        print ("data Found",smiles,sequence,interaction_probability,binary_class)
        return redirect(url_for('search_item' , interaction=interaction_probability ,substrate=substrate,smiles=smiles ))
    return render_template('search.html' , form=form)



@app.route('/search')
def search_item():
    substrate = request.args.get('substrate',default=None) 
    interaction = request.args.get('interaction',default=None) 
    smiles = request.args.get('smiles',default=None) 

    return render_template('result.html',smiles=smiles , interaction=interaction ,substrate=substrate,test = "nothingWorks")

@app.route('/analysis')
def analysis():
    return render_template('_base.html')

@app.route('/tutorial')
def tutorial():
    return render_template('_base.html')

@app.route('/download')
def download():
    return render_template('_base.html')


@app.route('/about')
def about():
    return render_template('_base.html')


@app.route('/contact')
def contact():
    return render_template('_base.html')