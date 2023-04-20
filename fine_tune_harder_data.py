# fine_tune_class_1.py
# fine tune the Robeta model2.1 for classification:


# set GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# import
import copy
import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

from transformers import RobertaConfig
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


# load previous model
model_path = "EsperBERTo"

config = RobertaConfig(
        vocab_size=3000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=10,
        type_vocab_size=1,
        num_labels=4
        )


# load tokenizer
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)


# load model
from transformers import RobertaForMaskedLM
from transformers import RobertaForSequenceClassification

model1 = RobertaForMaskedLM.from_pretrained(model_path)
model2 = RobertaForSequenceClassification(config)
#model2 = RobertaForSequenceClassification.from_pretrained(model2, num_labels=4, problem_type="multi_label_classification")
model2.roberta = copy.deepcopy(model1.roberta)
del model1
torch.cuda.empty_cache()

# tokenizer 
# tokenizer = ByteLevelBPETokenizer(model_path+"/vocab.json", model_path+"/merges.txt")

# tokenizer._tokenizer.post_processor = BertProcessing(
#         ("</s>", tokenizer.token_to_id("</s>")),
#         ("<s>", tokenizer.token_to_id("<s>")),
#         )

# tokenizer.enable_truncation(max_length=512)

from transformers import PreTrainedTokenizerFast
# tokenizer.save("byte-level-BPE.tokenizer.json")
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# load dataset
from datasets import load_dataset
dataset_LIPO = load_dataset("text", data_dir="LIPO/train")
dataset_NO_SP = load_dataset("text", data_dir="NO_SP/train")
dataset_TAT = load_dataset("text", data_dir="TAT/train")
dataset_TATLIPO = load_dataset("text", data_dir="TATLIPO/train")


# add label
def add_label_0(d):
    d['labels'] = 0
    return d

dataset_LIPO = dataset_LIPO.map(add_label_0)

def add_label_1(d):
    d['labels'] = 1
    return d

dataset_NO_SP = dataset_NO_SP.map(add_label_1)

def add_label_2(d):
    d['labels'] = 2
    return d

dataset_TAT = dataset_TAT.map(add_label_2)

def add_label_3(d):
    d['labels'] = 3
    return d

dataset_TATLIPO = dataset_TATLIPO.map(add_label_3)


# build dataset
from datasets import concatenate_datasets
dataset = concatenate_datasets([dataset_LIPO['train'],dataset_NO_SP['train'],dataset_TAT['train'],dataset_TATLIPO['train']])
dataset = dataset.shuffle(seed=42)
def tokenization(e):
    return tokenizer(e["text"], padding=True, truncation=True)

dataset = dataset.map(tokenization, batched=True, batch_size = len(dataset))

#def del_text(e):
#    del e['text']
#    return e

#dataset = dataset.map(del_text, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# fine-tune
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
        output_dir="./Model_ft_harder_task",
        overwrite_output_dir=True,
        num_train_epochs=12,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        save_steps=5_000,
        save_total_limit=10,
        learning_rate=1e-4
        )

trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=dataset,
        )

trainer.train()
trainer.save_model("./Model_ft_harder_task")
