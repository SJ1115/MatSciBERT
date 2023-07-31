import sys
sys.path.insert(0,'..')

from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc, AbsDoc

import re
import csv
from tqdm import tqdm

from src.util import callpath
from config import config

## Get doi list
doi = []
with open(callpath("./data/pretrain/piis_dois.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for i, line in enumerate(reader):
        doi.append(line[1])

#print(len(doi))
## 153978

## Load configuration & Initialize client
class ClientReader:
    def __init__(self):
        self.clients = [ElsClient(key)
                for key in config.Elsevier_keys]
    def read(self, doc):
        for cli in self.clients:
            if doc.read(cli):
                break
            ## warning?
reader = ClientReader()

remove_tags = [
    r"© [0-9]+",
    r"©[0-9]+",
    "Elsevier B.V",
    "Elsevier Ltd",
    "Elsevier GmbH",
    "Elsevier Inc",
    "Elsevier Masson SAS",
    "and Techna Group S.r.l.",
    "Published by",
    "All rights reserved.",
    "Chinese Society of Rare Earths",
    "American Pharmacists Association®",
    "SECV",
    "The Society for Range Management.",
    "The Society of Powder Technology Japan"
]
fi, l = 0, 0
k = 0 
#### for fix
fi = 4
#### \for fix
f = open(callpath(f"data/pretrain/txt/abs_{fi}.txt"), 'w')
iterator = tqdm(total=len(doi))
for i, di in enumerate(doi):
    iterator.update()
    #### for fix
    if i <= 54797: 
        continue
    #### \for fix
    try:
        doi_doc = FullDoc(doi = di)
        reader.read(doi_doc)
        
        scp_id = int(doi_doc.data['link']['@href'].split("/")[-1])
        scp_doc = AbsDoc(scp_id = scp_id)
        reader.read(scp_doc)
        
        abstract = scp_doc.data['item']['bibrecord']['head']['abstracts']

        ## process
        for tag in remove_tags:
            abstract = re.sub(tag, "", abstract)
        
        abstract = abstract.strip()
        if abstract.startswith("."):
            abstract = abstract[1:]
        if abstract.endswith(" ."):
            abstract = abstract[:-2]
        f.write(abstract.strip()+"\n")
        l += 1
        iterator.set_description(f"{k} files done")
        k += 1
        
        if l >= 10000:
            f.close()
            fi += 1
            f = open(callpath(f"data/pretrain/txt/abs_{fi}.txt"), 'w')
            l = 0
    except:
        0
f.close()