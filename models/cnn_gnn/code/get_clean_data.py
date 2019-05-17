from collections import defaultdict
from rdkit import Chem
import numpy as np

def create_atoms(mol):
    # NOTE: my error handling
    try:
        atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    except Exception as e:
        print "Error creating atoms: {}".format(str(e))
        return None
    return np.array(atoms)




if __name__ == "__main__":
    with open('../dataset/soha_new_neg/original/'
                                        'composite_pos_neg_interactions.txt', 'r') as f:
        cpi_list = f.read().strip().split('\n')

    """Exclude data contains "." in the smiles."""
    cpi_list = list(filter(lambda x:
                           '.' not in x.strip().split()[0], cpi_list))
    N = len(cpi_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    word_dict = defaultdict(lambda: len(word_dict))

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
                print "CPI line did not have a third element; setting -999 as sentinel"
                interaction = -999

            # if len(cpi_data) == 3:
            #     smiles, sequence, interaction = cpi.strip().split()
            # elif len(cpi_data) == 2:
            #     smiles, sequence = cpi.strip().split()
            # else:
            #     raise Exception ("Unexpected input, CPI file line has {} elements: {}".format(len(cpi_data), cpi_data))

            mol = Chem.MolFromSmiles(smiles)
            atoms = create_atoms(mol)
            # NOTE: my error handling
            if atoms is None:
                print "failure in sequence no {}, {}".format(no, cpi)
                continue

            with open("composite_pos_neg_interactions_no_period_no_failure.txt", "a") as out:
                out.write(cpi + "\n")