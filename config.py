from dataclasses import dataclass

@ dataclass
class config:
    Elsevier_keys = [
        "", # Key 0
        "", # Key 1
        "", # Key 2
        "", # Key 3
        "", # Key 4
        "", # Key 5
        "", # Key 6
        "", # Key 7
        "", # Key 8
        "", # Key 9
    ]

    max_seq_length = 512
    mlm_prob = .15
    valid_ratio = .1
    # Set 0 means not to fix random seed.
    seed = 15
    batch_size = 16
    total_epoch = 30*15
    
    #### for task ####
    task_epoch = 5
    task_lr = 1e-3
    task_batch_size = 8
    task_betas = (.9, .997)
    task_eps = 1e-6
    task_w_decay = 1e-2
    # NER (SOFC)
    ner_epoch = 5
    ner_lr = 1e-3
    ner_batch_size = 8
    ner_betas = (.9, .997)
    ner_eps = 1e-6
    ner_w_decay = 1e-2

    
class config_NER(config):
    # NER (SOFC)
    task_epoch = 5
    task_lr = 1e-3
    task_batch_size = 8
    task_betas = (.9, .997)
    task_eps = 1e-6
    task_w_decay = 1e-2

class config_QA(config):
    # QA (ScienceQA)
    task_epoch = 20
    task_lr = 1e-5
    task_batch_size = 16
    task_betas = (.9, .997)
    task_eps = 1e-6
    task_w_decay = 1e-2