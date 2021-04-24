"""transformers의 Trainer 활용 시 추론에 사용하는 모듈"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import REDatasetForTrainer, load_data
from models import load_model
from tokenization import load_tokenizer, tokenize
from preprocessing import preprocess_text
from config import *



def inference(model: nn.Module, tokenized_sent, device: str):
    """모델에 토큰화된 데이터를 입력, 추론 결과를 리턴하는 함수.
    추론 결과를 앙상블에 활용할 수 있으므로 기본적으로 soft label 형태로 리턴

    Args:
        model (torch.nn): 사전 학습된 모델
        tokenized_sent: 토크나이저를 통해 토큰화된 데이터
        device ([type]): CPU 또는 GPU
    """    
    dataloader = DataLoader(tokenized_sent, batch_size=512, shuffle=False, drop_last=False)
    model.eval()
    
    pred_list = []
    
    with torch.no_grad():
        for data in dataloader:
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
#                 token_type_ids=data['token_type_ids'].to(device), # BERT 아키텍쳐를 사용할 경우 활성화 TODO: 촌스럽다. 알아서 돌아가도록 해보자
                ).logits
            
            soft_preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            pred_list.append(soft_preds)
    
    pred_arr = np.vstack(pred_list)
    
    return pred_arr


def get_model_type(model_dir: str):
    model_type_raw = model_dir.split('/')[2]
    model_type = model_type_raw.split('_')[0]
    return model_type


def main(args):
    device = Config.Device

    # load tokenizer
    tokenizer = load_tokenizer(model_type=args.model_type, preprocess_type=args.preprocess_type)

    # load test datset
    data = load_data(args.test_dir)
    data = preprocess_text(data, args.model_type, args.preprocess_type)
    labels = data['label'].values
    tokenized_data = tokenize(data, tokenizer)
    dataset = REDatasetForTrainer(tokenized_data, labels)
    
    # load model
    model_type = get_model_type(args.model_dir)
    model = load_model(model_type=model_type, load_state_dict=args.model_dir)
    model.to(device)
    
    # inference and export
    pred_arr = inference(model, dataset, device)
    soft_result = pd.DataFrame(pred_arr) # save as soft labels for ensemble
    soft_result.to_csv(args.save_name, index=False)
    print(f"file saved. {args.save_name}")



if __name__ == '__main__':
    MODEL_DIR = "./saved_models/XLMSequenceClfLarge_xlm-roberta-large_20210422163026/checkpoint-2200/"
    SAVE_NAME = './predictions/test_fold3kr_softpred_xlmroberta_large_seqclf-ensemble_multi.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    parser.add_argument('--test-dir', type=str, default=Config.TestKr)
    parser.add_argument('--model-type', type=str, default=ModelType.XLMSequenceClfL)
    parser.add_argument('--preprocess-type', type=str, default=PreProcessType.ES)
    parser.add_argument('--save-name', type=str, default=SAVE_NAME)
    args = parser.parse_args()
    print(args)
    main(args)