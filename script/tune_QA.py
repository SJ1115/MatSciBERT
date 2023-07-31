import sys, os
sys.path.insert(0,'..')

import pandas as pd
import numpy as np

from src.loader import load_model
from src.task.ScienceQA import Get_ScienceQA, Trainer_ScienceQA
from src.task.OpenBookQA import Get_OpenBookQA, Trainer_OpenBookQA

from config import config_QA as config

from argparse import ArgumentParser

parser = ArgumentParser() # "./data/pretrain/norm"
parser.add_argument('--benchmark', default="ScienceQA", required=True, type=str) 
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--times', default=5, type=int)
args = parser.parse_args()

qa_name = args.benchmark
device = args.device
model_name = args.model
times = args.times

if qa_name.lower() in ('scienceqa', 'science'):
    dataset = Get_ScienceQA()
    config.task_batch_size = 1
    config.task_epoch = 5
    Trainer = Trainer_ScienceQA
    
elif qa_name.lower() in ('openbookqa', "openbook"):
    dataset = Get_OpenBookQA()
    Trainer = Trainer_OpenBookQA


score_sheet = pd.DataFrame(
    columns = ['mean', 'std', 'max', 'lr', 'beta1', 'beta2', 'eps', 'decay']
)


for lr in [2e-4, 1e-5, 5e-7]:
    config.task_lr = lr
    for betas in [(.9, .997), (.9, .999)]:
        config.task_betas = betas
        for eps in [2e-4, 1e-5, 5e-7]:
            config.task_eps = eps
            for decay in [0, 1e-2, 5e-4]:
                config.task_w_decay = decay

                score = []
                for t in range(times):
                    model, tokenizer = load_model(model_name, 'MC')
                    
                    trainer = Trainer(model, tokenizer, device)
                    trainer.train(dataset['train'], config)
                    score.append(trainer.test(dataset['validation'], config))
                    
                    del model, tokenizer, trainer

                line = pd.DataFrame({
                    "mean":np.mean(score), 
                    "std":np.std(score),
                    "max":np.max(score), 
                    "lr":lr,
                    "beta1":betas[0],
                    "beta2":betas[1],
                    "eps":eps,
                    "decay":decay
                }, index=[0])
                score_sheet = pd.concat((score_sheet, line))

print(f"Benchmark <{qa_name.upper()}>\nFor model <{model_name.upper()}>")
print(score_sheet.sort_values(by='mean', ascending=False).head(5))