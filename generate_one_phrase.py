import time
import logging

from english.BERT.get_translate import translate
from english.BERT.choose_var import get_prep,get_art
from english.BERT.config import view_res

logging.getLogger().setLevel(logging.ERROR)


tags = input('tags: ')

tags = tags.strip('[ ]').split(',')
tags = [i.strip() for i in tags]
sentence = ' '.join(tags)
len_of_each_part = [len(tag) for tag in tags]

pos = -1

for part in range(len(tags) - 1):
    pos += len_of_each_part[part] + 1
    sentence = sentence[:pos+1] + '[MASK]' + sentence[pos:]

    predict, art = get_prep(sentence,view_res)
    # print(predict)

    if art:
        pos += len(art)+1
        predict = predict.strip()
        if predict.startswith('[SEP]'):
            sentence = predict[5:]
        if predict.endswith('[CLS]'):
            sentence = predict[:-5]
        else:
            sentence = predict
    else:
        pos += 1
        sentence = predict

predict = predict[0].lower() + predict[1:]
if predict.startswith('to'):
    predict = 'to ' + predict[2:]
else:
    predict = 'to ' + predict

print(predict)
# translate(predict)


