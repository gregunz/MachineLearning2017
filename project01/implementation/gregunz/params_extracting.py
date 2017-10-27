# -*- coding: utf-8 -*-
"""
Extracting parameters from data files 
"""

import csv
from glob import glob

def extract_data(filename, is_combinations=False):
    scores = []
    pairs = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar="|")
        for idx, row in enumerate(spamreader):
            if(idx == 0):
                scores = [float(e) for e in row]
                if(not is_combinations):
                    return scores

            if(idx == 2):
                ls = [int(e.replace("\"", "").replace("(", "").replace(")", ""))  for e in row]
                pairs = list(zip(ls[0::2], ls[1::2]))
    return scores, pairs

def filenames_combinations(dataname, l):
    return [glob("data/tuning/best_{d}/*_{i}.csv".format(d=dataname, i=i))[0] for i in range(l)] #[0] = taking only first filename

def filenames_xs_mask(dataname, l):
    return [glob("data/tuning/{d}/*_{i}.csv".format(d=dataname, i=i))[1] for i in range(l)] #[0] = taking only first filename

def only_above(limit, scores, values):
    return [value for score, value in zip(scores, values) if score > limit]

def build_combinations(dataname, limit, l):
    data = [extract_data(n, is_combinations=True) for n in filenames_combinations(dataname, l)]
    return [only_above(limit, scores, pairs) for scores, pairs in data]

def build_xs_mask(dataname, limit, l):
    data = [(extract_data(n), list(range(len(extract_data(n))))) for n in filenames_xs_mask(dataname, l)]
    return [only_above(limit, scores, indices) for scores, indices in data]
