
import logging
import sys
import timeit
import torch
import os
from rdkit import Chem
from configparser import ConfigParser
from models.cnn_gnn.code.Preprocess_data import (
    Split_sequence, load_dictionary,
    create_adjacency, create_atoms,
    create_fingerprints, create_ijbonddict,
)

from models.cnn_gnn.code.run_training import (
    Predictor, CompoundProteinInteractionPrediction
)

def setup_custom_logger(name):
	formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
								  datefmt='%Y-%m-%d %H:%M:%S')
	handler = logging.FileHandler('log.txt', mode='w')
	handler.setFormatter(formatter)
	screen_handler = logging.StreamHandler(stream=sys.stdout)
	screen_handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(handler)
	logger.addHandler(screen_handler)
	return logger

logger = setup_custom_logger('myapp')


def parseParams(path, param):
    parser = ConfigParser()
    parser.read(path)
    return parser.get('parameters', param).replace("'" ,'')

DATASET=parseParams('parameters.cnf','DATASET')#'human'
radius=parseParams('parameters.cnf','radius')#2
ngram=parseParams('parameters.cnf','ngram')#3
dim=parseParams('parameters.cnf','dim')#10
layer_gnn=parseParams('parameters.cnf','layer_gnn')#3
window=parseParams('parameters.cnf','window')#5
layer_cnn=parseParams('parameters.cnf','layer_cnn')#3
lr=parseParams('parameters.cnf','lr')#1e-4
lr_decay=parseParams('parameters.cnf','lr_decay')#0.5
decay_interval=parseParams('parameters.cnf','decay_interval')#10
iteration=parseParams('parameters.cnf','iteration')#100
setting=str(DATASET)+"--radius"+str(radius)+"--ngram"+str(ngram)+"--dim"+str(dim)+\
	"--layer_gnn"+str(layer_gnn)+"--window"+str(window)+"--layer_cnn"+str(layer_cnn)+\
	"--lr"+str(lr)+"--lr_decay"+str(lr_decay)+"--decay_interval"+str(decay_interval)

(dim, layer_gnn, window,
 layer_cnn) = map(int, [dim, layer_gnn, window, layer_cnn])

if torch.cuda.is_available():
	device = torch.device('cuda')
	logger.info('The code uses GPU...')
else:
	device = torch.device('cpu')
	logger.info('The code uses CPU!!!')

logger.info('Loading data...')

crntFolder = os.getcwd()
dir_input = (crntFolder +'/dataset/' + str(DATASET) + '/input/'
			 'radius' + str(radius) + '_ngram' + str(ngram) + '/')

atom_dict = load_dictionary(dir_input + 'atom_dict.pickle')
bond_dict = load_dictionary(dir_input + 'bond_dict.pickle')
fingerprint_dict = load_dictionary(dir_input + 'fingerprint_dict.pickle')
word_dict = load_dictionary(dir_input + 'word_dict.pickle')
n_fingerprint = len(fingerprint_dict) + 1
n_word = len(word_dict) + 1

radius, ngram = map(int, [radius, ngram])


def callPredictionMain(smiles, sequence):
	logger.info('Creating data...')

	Compounds, Adjacencies, Proteins, Interactions = [], [], [], []
	smiles_sequence_list = []

	#smiles, sequence = cp.strip().split()
	smiles_sequence_list.append((smiles, sequence))

	mol = Chem.MolFromSmiles(smiles)

	atoms = create_atoms(atom_dict, mol)
	i_jbond_dict = create_ijbonddict(bond_dict, mol)

	fingerprints = create_fingerprints(fingerprint_dict, atoms, i_jbond_dict, radius)
	Compounds.append(torch.LongTensor(fingerprints).to(device))

	adjacency = create_adjacency(mol)
	Adjacencies.append(torch.FloatTensor(adjacency).to(device))

	words = Split_sequence(word_dict, sequence, ngram)
	Proteins.append(torch.LongTensor(words).to(device))

	dataset = list(zip(Compounds, Adjacencies, Proteins))

	logger.info('Predictiing CPI...')

	model = CompoundProteinInteractionPrediction(n_fingerprint, dim, layer_gnn, window,layer_cnn, n_word).to(device)
	predictor = Predictor(model, setting)

	start = timeit.default_timer()
	smiles,sequence,interaction_probability,binary_class = predictor.predict(dataset, smiles_sequence_list)
	end = timeit.default_timer()
	time = end - start
	logger.info('Prediction has finished in ' + str(time) + ' sec!')
	return smiles,sequence,interaction_probability,binary_class
