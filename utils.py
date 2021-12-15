import os
from os.path import exists, join
import json

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_data_path(mode, encoder, lang):
    paths = {}
    if mode == 'train' and lang == 'en':
        paths['train'] = 'data/train_CNNDM_' + encoder + '.jsonl'
        paths['val']   = 'data/val_CNNDM_' + encoder + '.jsonl'
    elif mode == 'test' and lang == 'en':
        paths['test']  = 'data/test_CNNDM_' + encoder + '.jsonl'
    return paths

def get_result_path(save_path, cur_model):
    save_path = save_path[:-len(save_path.split("/")[-1])] # TEMPORARY
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path
