import transformers
from transformers import AutoModel
model = AutoModel.from_pretrained('model_ps_1',local_files_only=True)

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

paths = 'train_dataset_primary.txt'

tokenizer = ByteLevelBPETokenizer(
    "model_ps_1/vocab.json",
    "model_ps_1/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="model_ps_1",
    tokenizer="model_ps_1"
)

import random
import numpy as np
import copy
file = open(paths,mode='r')
content = file.read()
file.close() 
tokens = tokenizer.encode(content).tokens
tokens = tokens[1:-1]
tk = copy.deepcopy(tokens)
score = []
for i in range(0, int(len(tokens)/20)):
    r = int(random.randrange(0, len(tokens)))
    tokens[r] = "<mask>"
    masked_content = "".join(tokens)
    score.append(fill_mask(masked_content)[0]['score'])
    print(np.mean(score))
    print(i)
    tokens = copy.deepcopy(tk)
