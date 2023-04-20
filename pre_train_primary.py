#
# pre_train_primary.py - pre train with primary sequence
#
# Jeff - Mar, 2023

# Import
import os
from os.path import exists
import biolib
from Bio import SeqIO

input_file = "swiss_prot_bacteria_june_2022_processed.fasta"
data = list(SeqIO.parse(input_file, "fasta"))
# primary_sequence = [x.seq.__str__() for x in data]
primary_sequence = "\n".join(x.seq.__str__() for x in data)
with open('train_dataset_primary.txt', 'w') as f:
    f.write(primary_sequence)
   
