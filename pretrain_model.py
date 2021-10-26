import time
import logging
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer
)
import torch
import torch.nn as nn
import string

from english.BERT.config import model_path




class BERTPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')


    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,labels=labels)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
new_model = BERTPred()
path = model_path
new_model.load_state_dict(torch.load(path))
new_model.eval()



def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx



def get_all_predictions(text_sentence, top_clean=5, view_res=False):
    # ========================= BERT =================================
    top_k = 5
    input_ids, mask_idx = encode(tokenizer, text_sentence)
    with torch.no_grad():
        results = new_model(input_ids)
    predict = results[0]

    probs = torch.nn.functional.softmax(predict[0, mask_idx], dim=-1)
    # print(probs)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    model_outputs = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        model_outputs.append([predicted_token,f'{token_weight.tolist():.3f}'])

    if view_res:
        print(text_sentence)
        for i in model_outputs:
            print(i)
        print('-----')

    bert = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'bert': bert}, model_outputs



if __name__ == '__main__':
    text_sentence = 'Hold [MASK] angle tooth'
    get_all_predictions(text_sentence,  view_res=True)