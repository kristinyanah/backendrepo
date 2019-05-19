from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem



def load_dictionary(file_name):
    with open(file_name, 'rb') as f:
        d = pickle.load(f)
    dictionary = defaultdict(lambda: len(d))
    dictionary.update(d)
    return dictionary


def create_atoms(atom_dict, mol):
    # NOTE: my error handling
    try:
        atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    except Exception as e:
        print("Error creating atoms: {}".format(str(e)))
        return None
    return np.array(atoms)


def create_ijbonddict(bond_dict, mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(fingerprint_dict, atoms, i_jbond_dict, radius):
    """Extract r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(word_dict, sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":

    DATASET, radius, ngram, test = sys.argv[1:]
    radius, ngram = map(int, [radius, ngram])
    # make boolean
    test = test.lower() == 'true'

# TODO: replace this so it isn't hardcoded
#     with open('../dataset/' + DATASET + '/original/'
#               'smiles_sequence_interaction.txt', 'r') as f:
#         cpi_list = f.read().strip().split('\n')

    # if we're generating test data, pull from test set
    if test:
        testset_name = "comp_seq_list_C1013_S1"
        with open('../dataset/' + DATASET + '/test_original/'
                  + testset_name + '.txt', 'r') as f:
            cpi_list = f.read().strip().split('\n')
        # with open('../dataset/' + DATASET + '/test_original/'
        #           'comp_seq_list_C1013_S2.txt', 'r') as f:
        #     cpi_list = f.read().strip().split('\n')
    else:
            with open('../dataset/' + DATASET + '/original/'
                      '50_pos_50_neg_composite_interactions_no_period_no_failure.txt', 'r') as f:
                cpi_list = f.read().strip().split('\n')

    """Exclude data contains "." in the smiles."""
    cpi_list = list(filter(lambda x:
                    '.' not in x.strip().split()[0], cpi_list))
    N = len(cpi_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    Compounds, Adjacencies, Proteins, Interactions = [], [], [], []

    for no, cpi in enumerate(cpi_list):

        print('/'.join(map(str, [no+1, N])))

        # TODO: make this nicer (perhaps we unpack first two, then third is in try/except block where we pass if we except)
        # check for cpi data interaction
        # has_interaction = False
        cpi_data = cpi.strip().split()
        smiles = cpi_data[0]
        sequence = cpi_data[1]
        try:
            interaction = cpi_data[2]
        except:
            print("CPI line did not have a third element; setting -999 as sentinel")
            interaction = -999

        # if len(cpi_data) == 3:
        #     smiles, sequence, interaction = cpi.strip().split()
        # elif len(cpi_data) == 2:
        #     smiles, sequence = cpi.strip().split()
        # else:
        #     raise Exception ("Unexpected input, CPI file line has {} elements: {}".format(len(cpi_data), cpi_data))

        mol = Chem.MolFromSmiles(smiles)
        atoms = create_atoms(atom_dict, mol)
        # NOTE: my error handling
        if atoms is None:
            print("failure in sequence no {}, {}".format(no, cpi))
            continue

        i_jbond_dict = create_ijbonddict(bond_dict, mol)

        fingerprints = create_fingerprints(fingerprint_dict, atoms, i_jbond_dict, radius)
        Compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        Adjacencies.append(adjacency)

        words = split_sequence(word_dict, sequence, ngram)
        Proteins.append(words)

        interaction = np.array([int(interaction)])
        Interactions.append(interaction)
    # change dir name according to whether or not this is a test set
    if test:
        dir_input = ('../dataset/' + DATASET + '/test_input/'
                                               'radius' + str(radius) + '_ngram' + str(ngram) + '/' + testset_name + '/')
    else:
        dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/')
    # NOTE: this is a python3 thing, so doing it in python2
    # os.makedirs(dir_input, exist_ok=True)
    try:
        os.makedirs(dir_input)
    except:
        pass

    np.save(dir_input + 'compounds', Compounds)
    np.save(dir_input + 'adjacencies', Adjacencies)
    np.save(dir_input + 'proteins', Proteins)
    np.save(dir_input + 'interactions', Interactions)

    dump_dictionary(atom_dict, dir_input + 'atom_dict.pickle')
    dump_dictionary(bond_dict, dir_input + 'bond_dict.pickle')
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')
