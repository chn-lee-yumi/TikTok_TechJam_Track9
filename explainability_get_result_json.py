import copy
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import *
from transformers import BertTokenizer

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
labels = ["normal", "offensive", "hatespeech"]
target_groups = ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous", "None"]
label_to_index = {label: idx for idx, label in enumerate(labels)}
group_to_index = {group: idx for idx, group in enumerate(target_groups)}
num_labels_task1 = len(labels)
num_labels_task2 = len(target_groups)
device = torch.device("cuda")  # cuda mps cpu


def convert_data(test_data, list_dict, rational_present=True, topk=2):
    # https://github.com/hate-alert/HateXplain/blob/master/Preprocess/dataCollect.py
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""

    test_data_modified = copy.deepcopy(test_data)

    for post_id in test_data:
        text = test_data[post_id]['input_ids']
        attention = test_data[post_id]['rationales']
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text = []
        new_attention = []
        if (rational_present):
            new_attention = [0]
            new_text = [101]
            for i in range(len(text)):
                if (i in topk_indices):
                    new_text.append(text[i])
                    new_attention.append(attention[i])
            new_attention.append(0)
            new_text.append(102)
        else:
            for i in range(len(text)):
                if (i not in topk_indices):
                    new_text.append(text[i])
                    new_attention.append(attention[i])
        test_data_modified[post_id]['input_ids'] = new_text
        test_data_modified[post_id]['rationales'] = new_attention
        assert len(new_text) == len(new_attention)

    return test_data_modified


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def label_to_onehot(label):
    index = label_to_index[label]
    onehot = torch.zeros(len(label_to_index))
    onehot[index] = 1
    return onehot.to(device)


def group_to_onehot(group):
    index = group_to_index[group]
    onehot = torch.zeros(len(group_to_index))
    onehot[index] = 1
    return onehot.to(device)


class BertForMultiTaskClassificationAndTagging(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super(BertForMultiTaskClassificationAndTagging, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 32),
            nn.ReLU(),
        )  # classifier backbone
        self.classifier1 = nn.Sequential(
            nn.Linear(32, num_labels_task1),
            nn.Softmax(dim=1),
        )  # labels
        self.classifier2 = nn.Sequential(
            nn.Linear(32, num_labels_task2),
            nn.Softmax(dim=1),
        )  # target_groups

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output

        # Classification tasks
        mid = self.classifier(cls_output)
        logits_task1 = self.classifier1(mid)
        logits_task2 = self.classifier2(mid)

        # Use the last layer's attention for tagging
        # print(len(outputs.attentions))
        # print(outputs.attentions)
        # print(outputs.attentions[-1].shape)  # torch.Size([32, 12, 167, 167])
        last_layer_attention = outputs.attentions[-1]  # Shape: (batch_size, num_heads, seq_length, seq_length)
        attention_mean = torch.mean(last_layer_attention, dim=1)  # Average over heads, shape: (batch_size, seq_length, seq_length)
        attention_mean = attention_mean[:, 0, :]  # shape: (batch_size, seq_length)

        return logits_task1, logits_task2, attention_mean


def standaloneEval_with_rational(test_data, topk=2):
    test_dataset = HateDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    print("Running eval on test data...")
    true_labels_task1 = []
    pred_labels_task1 = []
    true_labels_task2 = []
    pred_labels_task2 = []
    logits_all_final_task1 = []
    logits_all_final_task2 = []
    attention_all = []
    input_mask_all = []
    post_id_all = []
    tqdm_dataloader = tqdm(test_dataloader, desc=f"Testing", unit="batch")
    for post_ids, input_ids, attention_mask, label, target_group, rationales in tqdm_dataloader:
        input_ids, attention_mask, label, target_group, rationales = \
            input_ids.to(device), attention_mask.to(device), label.to(device), target_group.to(device), rationales.to(device)

        logits_task1, logits_task2, probs_tagging = model(input_ids=input_ids, attention_mask=attention_mask)

        true_labels_task1.extend(torch.argmax(label, dim=1).tolist())
        pred_labels_task1.extend(torch.argmax(logits_task1, dim=1).tolist())
        true_labels_task2.extend(torch.argmax(target_group, dim=1).tolist())
        pred_labels_task2.extend(torch.argmax(logits_task2, dim=1).tolist())
        for logits in logits_task1:
            logits_all_final_task1.append(softmax(logits.tolist()))
        for logits in logits_task2:
            logits_all_final_task2.append(softmax(logits.tolist()))
        attention_all.extend(probs_tagging.tolist())
        input_mask_all.extend(attention_mask.tolist())
        post_id_all.extend(post_ids)

    pred_labels = pred_labels_task1
    true_labels = true_labels_task1
    logits_all_final = logits_all_final_task1

    attention_vector_final = []
    for x, y in zip(attention_all, input_mask_all):
        temp = []
        for x_ele, y_ele in zip(x, y):
            if (y_ele == 1):
                temp.append(x_ele)
        attention_vector_final.append(temp)

    list_dict = []

    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final, pred_labels, true_labels):
        if (ground_truth == 0):  # 0=normal
            continue

        temp = {}
        pred_label = labels[pred]
        ground_label = labels[ground_truth]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        temp["classification_scores"] = {"hatespeech": logits[0], "normal": logits[1], "offensive": logits[2]}

        topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]

        temp_hard_rationales = []
        for ind in topk_indicies:
            temp_hard_rationales.append({'end_token': ind + 1, 'start_token': ind})

        temp["rationales"] = [{"docid": post_id,
                               "hard_rationale_predictions": temp_hard_rationales,
                               "soft_rationale_predictions": attention,
                               "truth": ground_truth}]
        list_dict.append(temp)

    return list_dict, test_data


def get_final_dict_with_rational(topk=5):
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    list_dict_org, test_data = standaloneEval_with_rational(test_data, topk=topk)
    test_data_with_rational = convert_data(test_data, list_dict_org, rational_present=True, topk=topk)
    list_dict_with_rational, _ = standaloneEval_with_rational(test_data_with_rational, topk=topk)
    test_data_without_rational = convert_data(test_data, list_dict_org, rational_present=False, topk=topk)
    list_dict_without_rational, _ = standaloneEval_with_rational(test_data_without_rational, topk=topk)
    final_list_dict = []
    for ele1, ele2, ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
        ele1['sufficiency_classification_scores'] = ele2['classification_scores']
        ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
        final_list_dict.append(ele1)
    return final_list_dict


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def collate_fn(batch):
    # 因为多了post id
    post_ids = [item[0] for item in batch]
    input_ids = [item[0 + 1] for item in batch]
    attention_masks = [item[1 + 1] for item in batch]
    labels = torch.stack([item[2 + 1] for item in batch])
    target_groups = torch.stack([item[3 + 1] for item in batch])
    rationales = [item[4 + 1] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_rationales = pad_sequence(rationales, batch_first=True, padding_value=0)
    if padded_input_ids.shape != padded_rationales.shape:
        print(padded_input_ids)
        print(padded_rationales)
    return post_ids, padded_input_ids, padded_attention_masks, labels, target_groups, padded_rationales


class HateDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.ids = list(data_dict.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = self.data[self.ids[idx]]
        post_id = sample['id']  # 注意这里和其它代码不同
        input_ids = torch.tensor(sample['input_ids']).to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)
        label = label_to_onehot(sample['label'])
        target_group = group_to_onehot(sample['target_group'])
        rationales = torch.tensor(sample['rationales']).to(device)

        return post_id, input_ids, attention_mask, label, target_group, rationales


model = BertForMultiTaskClassificationAndTagging(model_name, num_labels_task1, num_labels_task2).to(device)
model.load_state_dict(torch.load('model_4.pth'))
model.eval()

final_list_dict = get_final_dict_with_rational(topk=5)

with open('./explainability_data/results.json', 'w') as fp:
    fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))
