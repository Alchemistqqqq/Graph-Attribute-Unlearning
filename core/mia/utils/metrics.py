import torch
from sklearn.metrics import accuracy_score, confusion_matrix


def mia_metrics(pred, true, suffix=None):

    if suffix is None:
        suffix = ""
    else:
        suffix = suffix+"_"

    out_dict = dict()
    out_dict[suffix+"accuracy"] = accuracy_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    

    if tp+fn == 0:
        tpr = 0
    else:
        tpr = tp/(tp+fn)
    
    if fp + tn == 0:
        fpr = 0
    else:
        fpr = fp/(fp+tn)

    out_dict[suffix+"tpr"] = tpr
    out_dict[suffix+"fpr"] = fpr

    return out_dict
