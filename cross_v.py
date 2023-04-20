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

# load model
model_path = "Model_ft"

from transformers import RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained(model_path)

# tokenizer 
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# load dataset
from datasets import load_dataset
dataset_0 = load_dataset("text", data_dir="non_terminase_eval")
dataset_1 = load_dataset("text", data_dir="terminase_eval")

# add label
def add_label_0(d):
    d['labels'] = 0
    return d

dataset_0 = dataset_0.map(add_label_0)

def add_label_1(d):
    d['labels'] = 1
    return d

dataset_1 = dataset_1.map(add_label_1)

# build dataset
from datasets import concatenate_datasets
dataset = concatenate_datasets([dataset_0['train'],dataset_1['train']])

def tokenization(e):
    return tokenizer(e["text"], padding=True, truncation=True)

dataset = dataset.map(tokenization, batched=True, batch_size = len(dataset))
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5, random_state=1, shuffle=True)

print(dataset)
import numpy as np
splits = kf.split(np.zeros(dataset.num_rows), dataset['labels'])
from datasets import DatasetDict
for train_idxs, val_idxs in splits:
    f_dataset = DatasetDict({
        "train":dataset.select(train_idxs),
        "validation":dataset.select(val_idxs)})
    
    # fine-tune
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
            output_dir="./Model_cv",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 16,
            save_steps=5_000,
            save_total_limit=10
            )

    from sklearn.metrics import classification_report

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
            
        print(classification_report(labels, predictions))
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
            model=model,
            args=training_args,
            #data_collator=data_collator,
            train_dataset=f_dataset["train"],
            eval_dataset=f_dataset["validation"],
            compute_metrics = compute_metrics
            )

    trainer.train()
    trainer.evaluate(f_dataset["validation"])
    trainer.save_model("./Model_cv")
