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
model_path = "Model_ft_harder_task"

from transformers import RobertaForSequenceClassification
#model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=4)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=4, problem_type="multi_label_classification")

# data path
paths = [str(x) for x in Path("TAT/test").glob("*.txt")]

# tokenizer 
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("test for LIPO")
# validation
total = 0
correct = 0
for i in range(0, len(paths)):
    total += 1
    f = open(paths[i],mode='r')
    content = f.read()
    f.close()
    tokens = tokenizer(content, return_tensors="pt")
    #print(tokens)
    with torch.no_grad():
        logits = model(**tokens).logits
        predicted_class_id = logits.argmax().item()
        #print(model.config.id2label[predicted_class_id])
        if predicted_class_id == 2:
            correct += 1
        else:
            print("incorrect prediction:")
            print(content)


print("total:")
print(total)
print("correct:")
print(correct)
