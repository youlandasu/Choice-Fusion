# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

import json
from sklearn.metrics import precision_recall_fscore_support, f1_score
import numpy as np
import torch
import tqdm
import torchvision.transforms as transforms
from copy import deepcopy
from data_loader import InputFeatures, InputExample
# Strict match evaluation from https://github.com/jasonwu0731/trade-dst/blob/master/models/TRADE.py
# check utils/prediction_sample.json for the format of predictions
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def evaluate_metrics(all_prediction, SLOT_LIST):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for idx, dial in all_prediction.items():
        for k, cv in dial["turns"].items():
            if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                joint_acc += 1

            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv["pred_belief"]), SLOT_LIST)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv["pred_belief"]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total!=0 else 0
    turn_acc_score = turn_acc / float(total) if total!=0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
    return joint_acc_score, F1_score, turn_acc_score
def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = simple_accuracy(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def f1_score_with_invalid(targets, predictions):
    """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.
    Args:
        targets: np.ndarray of targets, either 0 or 1
        predictions: np.ndarray of predictions, any integer value
    Returns:
        F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, set it to the opposite of what the target is
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * f1_score(targets, predictions)}

def multirc_f1_over_all_answers(targets, predictions):
    """Special metric for MultiRC which computes F1 score over all examples.
    This is necessary because the targets/predictions for MultiRC are dicts and
    the f1_score_with_invalid expects a list of True/False labels, not dicts. As
    a result we just need to key in the "value" for each of the example dicts
    before feeding into f1_score_with_invalid.
    Args:
        targets: list of dicts, where each dict has a "value" key.
        predictions: list of dicts, where each dict has a "value" key.
    Returns:
        F1 score over values, where any prediction != 0 or 1 is counted as wrong.
    """
    return f1_score_with_invalid(
        [t["value"] for t in targets], [p["value"] for p in predictions]
    )

def model_eval(model,dataset,device):
    all_preds = []
    loss = 0
    for i in tqdm.tqdm(range(0,len(dataset),500)):
        if i + 500 <= len(dataset):
            features = dataset[i:i+500]
        else:
            features = dataset[i:]
        batch = {}
        for field in features[0].__dataclass_fields__:
            batch[str(field)] = []
        del batch['label'] 
        batch["labels"] = []
        for feature in features:
            for field in feature.__dataclass_fields__:
                value = getattr(feature, field)

                if isinstance(value, str):
                    batch[str(field)].append(value)
                elif value is None:
                    batch[str(field)]=None

                elif isinstance(value, int):
                    value = [value]
                    v = np.array(value,dtype=int)
                    batch["labels"].append(torch.LongTensor(v))
                else:
                    tensor_opts = [torch.tensor(option[0]).unsqueeze(0) for option in value]
                    tensor_alls = torch.cat(tensor_opts,dim=0)
                    batch[str(field)].append(tensor_alls.unsqueeze(0))

        del batch['example_id'] # This key is unexpected
        del batch['token_type_ids'] 
        del batch['token_type_ids2'] 
        for key,value in batch.items():
            if key in ["input_ids","attention_mask",'input_ids2','attention_mask2',"labels"]:
                batch[key] = torch.cat(batch[key],dim=0)
        eval_inputs = {k: v.to(device) for k,v in batch.items()}
        
        with torch.no_grad():
            output = model(**eval_inputs)
        logits = output.logits.cpu().numpy()
        loss += output.loss.cpu().numpy()
        preds = np.argmax(logits, axis=1)
        all_preds+=preds.tolist()

    all_preds=np.array(all_preds)
    #print(all_preds)
    ids = []
    for feature in dataset:
        truth = getattr(feature, "label")
        ids.append(truth)
    label_ids = np.array(ids)
    #print(label_ids)
    return (all_preds == label_ids).mean(), loss

def feature2dict_tensor(features):
    batches = []

    for idx in range(len(features)):
        batch={}
        for field in InputFeatures.__dataclass_fields__:
            value = getattr(features[idx], field)
            batch[str(field)]=torch.LongTensor(value)
        batches.append(batch)

    return batches #list of dicts

def example2dict(examples):
    batches = []
    for example in examples:
        batch={}
        for field in InputExample.__dataclass_fields__:
            value = getattr(example, field)
            batch[str(field)]=value
        for field in InputFeatures.__dataclass_fields__:
            value = getattr(example, field)
            batch[str(field)]=value
        batches.append(batch)
        assert len(batch) == len(batches[0])
    return batches #list of dicts

def bundlefeature2dict(features):
    batches = []
    for feature in features:
        batch={}
        for field in feature.__dataclass_fields__:
            value = getattr(feature, field)
            if isinstance(value,str):
                batch[str(field)]=value
            else:
                batch[str(field)]=torch.LongTensor(value)
        batches.append(batch)
        assert len(batch) == len(batches[0])
    return batches #list of dicts



if __name__ == "__main__":
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    with open("save/t5/results/zeroshot_prediction.json") as f:
        prediction = json.load(f)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(prediction, ontology)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(evaluation_metrics)
