import sys, os
sys.path.insert(0,'..')

from src.loader import load_model, MSC_Dataset, data_collator
from src.util import callpath
from config import config

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
#import torch
from argparse import ArgumentParser


parser = ArgumentParser() # "./data/pretrain/norm"
parser.add_argument('--train_dir', default=".", required=True, type=str) 
parser.add_argument('--save_dir', required=True, type=str)
parser.add_argument('--device', default="0", type=str)
args = parser.parse_args()

train_dir = callpath(args.train_dir)
save_dir = callpath("result/model/" + args.save_dir + "/save")
log_dir  = callpath("result/model/" + args.save_dir + "/logs")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.device

if config.seed:
    set_seed(config.seed)

model, tokenizer = load_model("SciBERT", task="MLM")

txt_list = []
for filename in os.listdir(train_dir):
    if filename.endswith(".txt"):
        txt_list.append(os.path.join(train_dir, filename))
assert len(txt_list) , "corpus file(.txt) dost NOT exist"
  
Corpus = MSC_Dataset(input_txt = txt_list, single_file = False)

train_set, valid_set = Corpus.train_valid_split(config.valid_ratio)

training_args = TrainingArguments(
    output_dir=save_dir,
    logging_dir=log_dir,
    report_to='tensorboard',
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    warmup_ratio=0.048,
    learning_rate=1e-4,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=0.0,
    num_train_epochs=config.total_epoch,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

train_res = trainer.train(resume_from_checkpoint=True)
print(train_res)