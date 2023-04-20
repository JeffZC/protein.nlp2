from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "EsperBERTo/vocab.json",
    "EsperBERTo/merges.txt",
)


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
torch.cuda.is_available()


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=3000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=10,
    type_vocab_size=1,
)


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)




from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)



from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="train_ds",
    block_size=128,
)



from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.20
)



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_gpu_train_batch_size=128,
    save_steps=50_000,
    save_total_limit=10,
    prediction_loss_only=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


trainer.train()
trainer.save_model("./EsperBERTo")

PATH = "./torch_model"
torch.save({
    'roBERTa_state_dict': model.roberta.state_dict(),
    'LM_state_dict': model.lm_head.state_dict()
    }, PATH)
