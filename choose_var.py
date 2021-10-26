from english.BERT.pretrain_model import get_all_predictions

prep = ['at','with','in','on','at','onto','into','about', 'above','of','by', 'for', 'from','over','through','via', 'since','after'] # добавить больше предлогов
articles = ['a','an','the']


def get_art(one_mask):
    return one_mask


def get_prep(sencond_mask,view_res):
    res,model_outputs = get_all_predictions(sencond_mask, view_res=view_res)
    # res = unmasker(sencond_mask)
    count = 0
    # print(res)

    for variants in model_outputs:  # учитываем пока что только 1 вариант предлога
        count += 1
        if ((variants[0] in prep) or (variants[0] in articles))and (float(variants[1]) > 0.1):
            predicked = sencond_mask.replace('[MASK]',str(variants[0]))
            return predicked, variants[0]
        if count > 4:
            count = 0
            return sencond_mask.replace('[MASK]', ''), False
    return sencond_mask.replace('[MASK]', ''),False