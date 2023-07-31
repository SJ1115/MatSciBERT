import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import tensor
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from torcheval.metrics.functional import (
    multiclass_f1_score as f1_score,
    multiclass_accuracy as accuracy
)

from datasets import load_dataset
from tokenizers.normalizers import BertNormalizer

from src.util import callpath
from src.task.base import BaseTrainer


__QA_name__ ="sofc_materials_articles"

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)
f = open(callpath('data/vocab_mappings.txt'))
mappings = f.read().strip().split('\n')
f.close()

NULL_ENTITY = 8
NULL_SLOT = 38

SOFC = load_dataset(__QA_name__)
#### Get DataSet
class SOFC_NER(Dataset):
	def __init__(self, dataset, tokenizer, is_slot=False):
		result = []

		if is_slot:
			pad_tok = NULL_SLOT
			lab_key = 'slot_labels'
		else:
			pad_tok = NULL_ENTITY
			lab_key = 'entity_labels'

		for item in dataset:
			sent_ind,  = np.where(item['sentence_labels'])
			sentences = [item['tokens'][i] for i in sent_ind]
			assert len(sentences) == len(item[lab_key])

			
			for tokens, labels in zip(sentences, item[lab_key]):
				## Norm and Map
				tokens = [norm.normalize_str(token) for token in tokens]
    
				for i, token in enumerate(tokens):
					for line in mappings:
						token = token.replace(line[0], line[2])
						tokens[i] = token
    
				## Tokenizer
				tokens = [tokenizer.encode(token)[1:-1] for token in tokens]
    
				## Fit length
				out_labels = [pad_tok]
				for i, token in enumerate(tokens):
					out_labels += [labels[i]] * len(token)
				out_labels += [pad_tok]
    
				## Flatten
				tokens = [item for sublist in tokens for item in sublist]
				tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]

				result.append({
					"input_ids" : tokens,
					"labels" : out_labels,
				})
		self.dataset = result

	def __len__(self, ):
		return len(self.dataset)

	def __getitem__(self, index):
		return self.dataset[index]

def get_SOFC_NER(tokenizer, is_slot=False):
	dataset = load_dataset(__QA_name__)
	dataset['train'] = SOFC_NER(dataset['train'], tokenizer, is_slot=is_slot)
	dataset['validation'] = SOFC_NER(dataset['validation'], tokenizer, is_slot=is_slot)
	dataset['test'] = SOFC_NER(dataset['test'], tokenizer, is_slot=is_slot)
	return dataset

#### Build Collate Function
class _collate:
    def __init__(self, padding=8, device='cpu'):
        self.padding = padding
        self.device = device
    
    def __call__(self, batch):
		# re-tie batch by each key
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = pad_sequence([tensor(item[key]) for item in batch],
                        batch_first=True, padding_value=self.padding).to(self.device)
		
        mask = torch.zeros_like(collated_batch[key]).to(self.device)
        for i, item in enumerate(batch):
            mask[i, :len(item[key])] = 1
        collated_batch['attention_mask'] = mask
        return collated_batch
    
#### Trainer ####
class Trainer_SOFO_NER(BaseTrainer):
	def __init__(self, model, tokenizer, device, is_slot=False):
		self.model = model.to(device)
		self.tokenizer = tokenizer
		self.device = device
		self.is_slot = is_slot

		self.loader = None

	"""def train(self, trainset:Dataset, config):
		self._load(trainset, config.ner_batch_size, True)

		optimizer = Adam(self.model.parameters(),
        	lr=config.ner_lr,
        	betas=config.ner_betas,
        	eps=config.ner_eps,
        	weight_decay=config.ner_w_decay)
      
		iterator = tqdm(total=self.loader.__len__()*config.ner_epoch)
		self.model.train()

		for _ in range(config.ner_epoch):
			for item in self.loader:
				output = self.model(**item)

				loss = output.loss
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

				iterator.update(1)
				iterator.set_description(f"loss:{loss.item():.3f}")"""
    
    
	def test(self, testset:Dataset, config):
		self._load(testset, config.ner_batch_size, False)

		self.model.eval()

		predict, answer = [], []

		with torch.no_grad():
			for item in tqdm(self.loader):
				output = self.model(**item)

				pred = output.logits.argmax(2)

				predict += [p.item() for line_pred, line_mask in zip(pred, item['attention_mask'])
											for p, m in zip(line_pred, line_mask) if m>0]
				answer  += [l.item() for line_labs, line_mask in zip(item['labels'], item['attention_mask'])
											for l, m in zip(line_labs, line_mask) if m>0]
		
		mac = f1_score(torch.tensor(predict), torch.tensor(answer),
               average = 'macro', num_classes = self.model.num_labels)
		mic = f1_score(torch.tensor(predict), torch.tensor(answer),
               average = 'micro')
		acc = accuracy(torch.tensor(predict), torch.tensor(answer),
               average = 'micro', num_classes = 9)

		return {"accuracy" : acc.item(), 
        		#"f1_macro" : mac.item(),
          		"f1_micro" : mic.item()}

	def _load(self, data, batch_size, is_train=True):
		padding = NULL_SLOT if self.is_slot else NULL_ENTITY
		
		self.loader = DataLoader(data, batch_size=batch_size, shuffle=is_train,
        	collate_fn=_collate(padding=padding, device=self.device))
		return