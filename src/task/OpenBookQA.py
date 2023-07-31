from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_f1_score as f1_score,
    multiclass_accuracy as accuracy
)

import pandas as pd
from tqdm import tqdm

from src.task.base import BaseTrainer

# "derek-thomas/ScienceQA"
__QA_name__ ="openbookqa"

#### Get DataSet
def Get_OpenBookQA():
    dataset = load_dataset(__QA_name__)

    dataset = dataset.flatten()
    
    dataset = dataset.rename_column("question_stem", "question")
    dataset = dataset.rename_column("choices.text", "choices")
    dataset = dataset.rename_column("answerKey", "answer")
    
    #dataset = dataset.remove_columns(['id'])
    dataset = dataset.remove_columns(['choices.label'])
        
    return dataset

#### Build Collate Function
to_int = {"A":0,"B":1,"C":2,"D":3}

class _collate:
    def __init__(self, tokenizer, device='cpu', show_id=False):
        self.tokenizer = tokenizer
        self.device = device
        self.show_id = show_id

    def __call__(self, batch):
        b = len(batch)
        c = len(batch[0]['choices'])
        
        questions = [item['question']
            for item in batch
                for _ in item['choices']]
        
        answers = [choice for item in batch for choice in item['choices']]

        inputs = self.tokenizer(questions, answers, return_tensors='pt', padding=True)

        labels = torch.tensor([to_int[item['answer']] for item in batch])
        
        inputs = {k: v.view(b, c, -1).to(self.device) for k, v in inputs.items()}
        inputs['labels'] = labels.to(self.device)
        
        if self.show_id:
            inputs = (inputs, [item['id'] for item in batch])
        return inputs

#### Trainer ####
class Trainer_OpenBookQA(BaseTrainer):
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.loader = None


	## train(self, dataset, config) comes from BaseTrainer.
    #### Refer base.py if needed.
    
    def test(self, testset:Dataset, config, result_csv:str=''):
        self._load(testset, batch_size=config.task_batch_size, is_train=False)

        result = pd.DataFrame(columns = ['ID', 'answer', 'predict'])

        self.model.eval()
        with torch.no_grad():
            for item in tqdm(self.loader):
                inputs, ids = item
                outputs = self.model(**inputs)
                answers = inputs['labels'].tolist()
                predicts = outputs.logits.argmax(-1).tolist()

                score = pd.DataFrame({'ID' : ids,
					'answer' : answers,
					'predict' : predicts
					})
                result = pd.concat((result, score))
            
            
        predict_all = torch.tensor(result.predict.tolist())
        answer_all = torch.tensor(result.answer.tolist())
        f_1 = f1_score(predict_all, answer_all, average='macro', num_classes=4)
        acc = accuracy(predict_all, answer_all)

        if len(result_csv):
            result.to_csv(result_csv, index=False)

            return  {"accuracy" : acc.item(), 
          		        "f1_macro" : f_1.item()}
        else:
            return acc.item()
    
    def _load(self, data, batch_size, is_train=True):
        self.loader = DataLoader(data, batch_size=batch_size, shuffle=is_train,
        	collate_fn=_collate(self.tokenizer, self.device, not is_train))
        return
      
   