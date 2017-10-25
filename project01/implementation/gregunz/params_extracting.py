# -*- coding: utf-8 -*-
"""
Extracting parameters from data files 
"""

import csv

def extract_data(filename):
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar="|")
        for idx, row in enumerate(spamreader):
            if(idx == 0):
                scores = [float(e) for e in row]

            if(idx == 2):
                ls = [int(e.replace("\"", "").replace("(", "").replace(")", ""))  for e in row]
                pairs = list(zip(ls[0::2], ls[1::2]))
    return scores, pairs

def filenames(dataname, l=6):
    return ["data/tuning/best_{d}/best_{d}_{idx}.csv".format(d=dataname, idx=i+1) for i in range(l)]

def only_above(v, scores, pairs):
    return [pair for score, pair in zip(scores, pairs) if score > v]

def build_combinations(dataname, limit):
    data = [extract_data(n) for n in filenames(dataname)]
    return [only_above(limit, scores, pairs) for scores, pairs in data]