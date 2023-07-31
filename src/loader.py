from transformers import (
    AutoConfig,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForTokenClassification,
    AutoTokenizer,
    DataCollatorForWholeWordMask
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from config import config

from src.util import random_split
##############################
#### Base Model : SciBERT ####
##############################

def load_model(model="BERT", task='MLM', num_labels=None):
    # SciBERT / MatSciBERT / BERT / Savepoint
    if model.lower() in ("scibert", 'sci', 'science') :
        model_name = 'allenai/scibert_scivocab_uncased'
    elif model.lower() in ('matscibert', 'mat', 'material', 'matsci'):
        model_name = "M3RG-IITD/MatSciBERT"
    elif model.lower() == 'bert':
        model_name = 'bert-base-uncased'
    elif model.lower() in ('bio', 'biobert'):
        model_name = "monologg/biobert_v1.1_pubmed"
    else: ## from Savepoint
        model_name = model

    if task.lower() == "mlm":
        model = BertForMaskedLM.from_pretrained(model_name,
            config = AutoConfig.from_pretrained(model_name))
    elif task.lower() == "mc":
        model = BertForMultipleChoice.from_pretrained(model_name,
            config = AutoConfig.from_pretrained(model_name))
    elif task.lower() == "ner":
        model = BertForTokenClassification.from_pretrained(model_name,
            config = AutoConfig.from_pretrained(model_name,
                num_labels=num_labels))
    else:
        raise Exception("Task is NOT Valid")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer



##########################################
#### Load tokenizer and Build Dataset ####
##########################################
# 1. Tokenizer 

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

start_tok = tokenizer.convert_tokens_to_ids('[CLS]')
sep_tok = tokenizer.convert_tokens_to_ids('[SEP]')
pad_tok = tokenizer.convert_tokens_to_ids('[PAD]')

# 2. File Reader

def file_sent_tokenize(file_name, max_seq_length=config.max_seq_length):
    f = open(file_name, 'r')
    sents = f.read().strip().split('\n')
    f.close()
    
    tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in tqdm(sents)]
    for s in tok_sents:
        s.pop(0)
    
    res = [[]]
    l_curr = 0
    
    for s in tok_sents:
        l_s = len(s)
        idx = 0
        while idx < l_s - 1:
            if l_curr == 0:
                res[-1].append(start_tok)
                l_curr = 1
            s_end = min(l_s, idx + max_seq_length - l_curr) - 1
            res[-1].extend(s[idx:s_end] + [sep_tok])
            idx = s_end
            if len(res[-1]) == max_seq_length:
                res.append([])
            l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}

def multi_file_tokenize(file_names, max_seq_length=config.max_seq_length):
    res = [[]]
    l_curr = 0
    
    for name in tqdm(file_names):
        f = open(name, 'r')
        sents = f.read().strip().split('\n')
        f.close()
        
        tok_sents = [tokenizer(s, padding=False, truncation=False)['input_ids'] for s in sents]
        for s in tok_sents:
            s.pop(0)
        
        
        
        for s in tok_sents:
            l_s = len(s)
            idx = 0
            while idx < l_s - 1:
                if l_curr == 0:
                    res[-1].append(start_tok)
                    l_curr = 1
                s_end = min(l_s, idx + max_seq_length - l_curr) - 1
                res[-1].extend(s[idx:s_end] + [sep_tok])
                idx = s_end
                if len(res[-1]) == max_seq_length:
                    res.append([])
                l_curr = len(res[-1])
    
    for s in res[:-1]:
        assert s[0] == start_tok and s[-1] == sep_tok
        assert len(s) == max_seq_length
        
    attention_mask = []
    for s in res:
        attention_mask.append([1] * len(s) + [0] * (max_seq_length - len(s)))
    
    return {'input_ids': res, 'attention_mask': attention_mask}

# 3. Dataset

class MSC_Dataset(Dataset):
    def __init__(self, input_txt, single_file=True, max_seq_length=config.max_seq_length):
        if single_file:
            self.inp = file_sent_tokenize(input_txt, max_seq_length)
        else:
            self.inp = multi_file_tokenize(input_txt, max_seq_length)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
        return item

    def __len__(self):
        return len(self.inp['input_ids'])
    
    def train_valid_split(self, valid_ratio=.1):
        train, valid = random_split(self,
            lengths=[1-valid_ratio, valid_ratio])
        return train, valid


data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm_probability=config.mlm_prob)

##############################
#### Fine Tuning for Task ####
##############################

#### 1. NER

#### 2. 

#### ++ Science QA
