from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset


class BaseTrainer:
	"""
	You have to build test() and _load() each time,
	according to each of dataset
 	"""
	def __init__(self, model, tokenizer, device):
		self.model = model.to(device)
		self.tokenizer = tokenizer
		self.device = device
		self.loader = None
    
	def train(self, trainset:Dataset, config):
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
				output = self.model(**item)

				loss = output.loss
				loss.backward()

				optimizer.step()
				optimizer.zero_grad()

				iterator.update(1)
				iterator.set_description(f"loss:{loss.item():.3f}")
    
	def test(self, ):
		0
    
	def _load(self, data:Dataset, batch_size, is_train=True):
		0