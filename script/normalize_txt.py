import sys
sys.path.insert(0,'..')
import random
from tqdm import tqdm
from argparse import ArgumentParser
from tokenizers.normalizers import BertNormalizer
from src.util import callpath


parser = ArgumentParser()
parser.add_argument('--in_file', required=True, type=str)
parser.add_argument('--out_norm_file', required=True, type=str)
args = parser.parse_args()


f = open(callpath('data/vocab_mappings.txt'))
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize_and_save(file_path, save_file_path):
    f = open(file_path)
    corpus = f.read().strip().split('\n')
    f.close()
    
    random.seed(42)
    corpus = [norm.normalize_str(sent) for sent in tqdm(corpus)]
    corpus_norm = []
    for sent in tqdm(corpus):
        norm_sent = ""
        for c in sent:
            if c in mappings:
                norm_sent += mappings[c]
            elif random.uniform(0, 1) < 0.3:
                norm_sent += c
            else:
                norm_sent += ' '
        corpus_norm.append(norm_sent)
    
    f = open(save_file_path, 'w')
    f.write('\n'.join(corpus_norm))
    f.close()


normalize_and_save(callpath(args.in_file), callpath(args.out_norm_file))
