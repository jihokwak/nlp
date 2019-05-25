import os
import numpy as np
import pandas as pd
from langdetect import detect
from glob import glob
import re
files = glob("02_seq2seq/data/Corpus10/*.txt")

pairs = []
for file in files :
    print(file)
    with open(file, "r") as f :
        text = f.read()
        pairs.extend(re.split(r"\[\d+\]", text))

prep_pairs = []
for pair in pairs:
    prep_pair = "\t".join(pair.split("\n")).replace("#","").strip()
    prep_pairs.append(prep_pair)
    print(prep_pair)

prep_pairs = prep_pairs[1:]

for p in prep_pairs :
    if len(p) < 2 :
        print(p)


save_file_name = "02_seq2seq/data/Corpus10/eng2kor.txt"
with open(save_file_name, "w", encoding='utf-8') as f:
    for prep_pair in prep_pairs :
        f.writelines(prep_pair+"\n")
        print(prep_pair)


prep_pairs[50000]