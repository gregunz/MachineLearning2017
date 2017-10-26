# -*- coding: utf-8 -*-
"""
Csv reading or writer helper functions
"""

import csv

def list_to_csv(ls, filename):
    with open(filename, "a", encoding="utf8") as output_file:
        wr = csv.writer(output_file, delimiter=",")
        wr.writerow(ls)