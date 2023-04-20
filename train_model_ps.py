from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

import torch

torch.cuda.is_available()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# file path
file_path = 'dataset_primary_train.txt'

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=file_path, vocab_size=5_000, min_frequency=10, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("model_ps_1")


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=5000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("model_ps_1", max_len=512)


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)



from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=128,
)



from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.20
)



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="model_ps_1",
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_gpu_train_batch_size=128,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


trainer.train()
trainer.save_model("model_ps_1")
