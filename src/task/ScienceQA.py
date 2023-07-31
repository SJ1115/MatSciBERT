from datasets import load_dataset, Dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from src.task.base import BaseTrainer

# "derek-thomas/ScienceQA"
__QA_name__ ="metaeval/ScienceQA_text_only"

#### Get DataSet
def Get_ScienceQA(train_filter=True):
	dataset = load_dataset(__QA_name__)

	if train_filter:
		dataset = dataset.filter(lambda x: x['subject']=='natural science')

	else:
		dataset['test'] = dataset['test'].filter(
        	lambda x: x['subject']=='natural science')
		dataset['validation'] = dataset['validation'].filter(
         	lambda x: x['subject']=='natural science')

	return dataset

#### Build Collate Function
def _collate(item, tokenizer, device='cpu'):
	"""
	It does not support batch, so batch_size=1 is needed.
	"""
	item = item[0]
	sentence = tokenizer(
		[item['question'] + item['hint']] * len(item['choices']), # 
		item['choices'],
		return_tensors='pt',
		padding=True)

	labels = torch.tensor(item['answer']).unsqueeze(0).to(device)
	inputs = {k: v.unsqueeze(0).to(device) for k,v in sentence.items()}
	grades = torch.tensor(int(item['grade'][-1])).to(device)
	topics = item['topic'] 
	return inputs, labels, grades, topics

#### Trainer ####
class Trainer_ScienceQA(BaseTrainer):
	def __init__(self, model, tokenizer, device):
		self.model = model.to(device)
		self.tokenizer = tokenizer
		self.device = device

		self.loader = None

	def train(self, trainset:Dataset, config, ):
		self._load(trainset, config.task_batch_size, True)
      
		optimizer = Adam(self.model.parameters(),
        	lr=config.task_lr,
        	betas=config.task_betas,
        	eps=config.task_eps,
        	weight_decay=config.task_w_decay)
      
		iterator = tqdm(total=self.loader.__len__()*config.task_epoch)
		self.model.train()
		
		for _ in range(config.task_epoch):
			for item in self.loader:
				data, label, _, _ = item
				
				output = self.model(**data, labels=label)

				loss = output.loss
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

				iterator.update(1)
				iterator.set_description(f"loss:{loss.item():.3f}")
    
    
	def test(self, testset:Dataset, config, result_csv:str=''):
		#assert len(result_csv),  "Filename for result.csv"
      
		self._load(testset, config.task_batch_size, False)
      
		result = pd.DataFrame(columns = ['grade', 'topic', 'match'])

		self.model.eval()
		with torch.no_grad():
			for item in tqdm(self.loader):
				data, label, grade, topic = item
				output = self.model(**data)
    
				pred = output.logits.argmax(-1).to("cpu")
				label = label.to("cpu")
    
				score = pd.DataFrame({'grade' : int(grade),
                    'topic' : topic,
					'match' : int(pred == label),
					}, index=[0])
				result = pd.concat((result, score))

		if len(result_csv):
			result.to_csv(result_csv, index=False)
		return result['match'].mean()
    
	def _load(self, data, batch_size, is_train=True):
		self.loader = DataLoader(data, batch_size=batch_size, shuffle=is_train,
        	collate_fn=lambda item: _collate(item, self.tokenizer, self.device))
		return
      
   