#!/usr/bin/env python3

import pickle

with open("/home/usman/Downloads/pythonCB/dataset/human/input/radius2_ngram3/word_dict.pickle", "rb") as f:
    w = pickle.load(f)

pickle.dump(w, open("word_dict.pickle","wb"), protocol=2)