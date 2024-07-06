import pickle
from collections import Counter

import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

labels = ["normal", "offensive", "hatespeech"]
target_groups = ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous", "None"]

label_to_index = {label: idx for idx, label in enumerate(labels)}
group_to_index = {group: idx for idx, group in enumerate(target_groups)}

device = torch.device("cuda")  # cuda mps cpu


def calculate_auc(y_true, y_pred, mask):
    """计算给定子群的AUC"""
    return roc_auc_score(y_true[mask], y_pred[mask])


def calculate_gmb_auc(auc_scores, p=-5):
    """计算广义均值偏差AUC（GMB AUC）"""
    return np.power(np.mean(np.power(auc_scores, p)), 1 / p)


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


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    target_groups = torch.stack([item[3] for item in batch])
    rationales = [item[4] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_rationales = pad_sequence(rationales, batch_first=True, padding_value=0)
    if padded_input_ids.shape != padded_rationales.shape:
        print(padded_input_ids)
        print(padded_rationales)
    return padded_input_ids, padded_attention_masks, labels, target_groups, padded_rationales


class HateDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.ids = list(data_dict.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = self.data[self.ids[idx]]
        input_ids = torch.tensor(sample['input_ids']).to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)
        label = label_to_onehot(sample['label'])
        target_group = group_to_onehot(sample['target_group'])
        rationales = torch.tensor(sample['rationales']).to(device)

        return input_ids, attention_mask, label, target_group, rationales


def get_target_group_distribution(data_dict):
    target_groups = [sample['target_group'] for sample in data_dict.values()]
    # target_groups = [sample['label'] for sample in data_dict.values()]
    return Counter(target_groups)


def calculate_weights(data_dict):
    target_groups = [sample['target_group'] for sample in data_dict.values()]
    # target_groups = [sample['label'] for sample in data_dict.values()]
    group_counts = Counter(target_groups)
    total_samples = len(target_groups)
    weights = {group: total_samples / count for group, count in group_counts.items()}
    sample_weights = [weights[sample['target_group']] for sample in data_dict.values()]
    # sample_weights = [weights[sample['label']] for sample in data_dict.values()]
    return sample_weights


def calculate_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_dataset = HateDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Model initialization
num_labels_task1 = len(labels)
num_labels_task2 = len(target_groups)
model = BertForMultiTaskClassificationAndTagging(model_name, num_labels_task1, num_labels_task2).to(device)

# Loss functions and optimizer
criterion_task1 = nn.CrossEntropyLoss()
criterion_task2 = nn.CrossEntropyLoss()
criterion_tagging = nn.BCELoss()

# Testing
model.load_state_dict(torch.load('model_4.pth'))
model.eval()
with torch.no_grad():
    total_loss_task1 = 0.0
    total_loss_task2 = 0.0
    total_loss_tagging = 0.0
    total_correct_task1 = 0
    total_correct_task2 = 0
    true_labels_task1 = []
    pred_labels_task1 = []
    true_labels_task2 = []
    pred_labels_task2 = []
    logits_all_final_task1 = []
    logits_all_final_task2 = []
    tqdm_dataloader = tqdm(test_dataloader, desc=f"Testing", unit="batch")
    for input_ids, attention_mask, label, target_group, rationales in tqdm_dataloader:
        input_ids, attention_mask, label, target_group, rationales = \
            input_ids.to(device), attention_mask.to(device), label.to(device), target_group.to(device), rationales.to(device)

        logits_task1, logits_task2, probs_tagging = model(input_ids=input_ids, attention_mask=attention_mask)
        # print(tokenizer.batch_decode(input_ids.tolist()))
        # print(input_ids)
        # print(logits_task1)
        # print(logits_task2)
        # print(probs_tagging)

        # Calculate losses
        loss_task1 = criterion_task1(logits_task1, torch.argmax(label, dim=1))
        loss_task2 = criterion_task2(logits_task2, torch.argmax(target_group, dim=1))
        loss_tagging = criterion_tagging(probs_tagging.reshape(-1), rationales.view(-1))

        total_loss_task1 += loss_task1.item()
        total_loss_task2 += loss_task2.item()
        total_loss_tagging += loss_tagging.item()

        # Calculate accuracy for task 1 and task 2
        total_correct_task1 += torch.sum(torch.argmax(logits_task1, dim=1) == torch.argmax(label, dim=1)).item()
        total_correct_task2 += torch.sum(torch.argmax(logits_task2, dim=1) == torch.argmax(target_group, dim=1)).item()

        true_labels_task1.extend(torch.argmax(label, dim=1).tolist())
        pred_labels_task1.extend(torch.argmax(logits_task1, dim=1).tolist())
        true_labels_task2.extend(torch.argmax(target_group, dim=1).tolist())
        pred_labels_task2.extend(torch.argmax(logits_task2, dim=1).tolist())
        for logits in logits_task1:
            logits_all_final_task1.append(softmax(logits.tolist()))
        for logits in logits_task2:
            logits_all_final_task2.append(softmax(logits.tolist()))

    # print(true_labels_task1)
    # print(pred_labels_task1)

    # Calculate average accuracy
    accuracy_task1 = total_correct_task1 / len(test_dataset)
    accuracy_task2 = total_correct_task2 / len(test_dataset)
    print(f"Test Accuracy - Task 1: {accuracy_task1:.4f}, Task 2: {accuracy_task2:.4f}, "
          f"Loss - Tagging: {total_loss_tagging / len(test_dataset):.4f}, "
          f"Task 1: {total_loss_task1 / len(test_dataset):.4f}, Task 2: {total_loss_task2 / len(test_dataset):.4f}"
          )

    print("===== Report for Task 1 =====")  # Paper use this as performance
    print(f"Accuracy Score of Task 1: {accuracy_score(true_labels_task1, pred_labels_task1)}")  # need this
    print(f"F1 Score of Task 1: {f1_score(true_labels_task1, pred_labels_task1, average='macro')}")  # need this
    # print(f"Precision Score of Task 1: {precision_score(true_labels_task1, pred_labels_task1, average='macro')}")
    # print(f"Recall Score of Task 1: {recall_score(true_labels_task1, pred_labels_task1, average='macro')}")
    print(f"ROCAUC Score of Task 1: {roc_auc_score(true_labels_task1, logits_all_final_task1, multi_class='ovo', average='macro')}")  # need this
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels_task1, pred_labels_task1))
    print(classification_report(true_labels_task1, pred_labels_task1, zero_division=0))

    print("===== Report for Task 2 =====")
    print(f"Accuracy Score of Task 2: {accuracy_score(true_labels_task2, pred_labels_task2)}")
    print(f"F1 Score of Task 2: {f1_score(true_labels_task2, pred_labels_task2, average='macro')}")
    # print(f"Precision Score of Task 2: {precision_score(true_labels_task2, pred_labels_task2, average='macro')}")
    # print(f"Recall Score of Task 2: {recall_score(true_labels_task2, pred_labels_task2, average='macro')}")
    print(f"ROCAUC Score of Task 2: {roc_auc_score(true_labels_task2, logits_all_final_task2, multi_class='ovo', average='macro')}")
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels_task2, pred_labels_task2))
    print(classification_report(true_labels_task2, pred_labels_task2, zero_division=0))

    print("===== Bias =====")
    true_labels_task1 = np.array(true_labels_task1)
    true_labels_task2 = np.array(true_labels_task2)
    logits_all_final_task1 = np.array(logits_all_final_task1)
    # 创建二进制标签
    binary_labels = np.where(true_labels_task1 == 0, 0, 1)
    # 将logits转换为概率
    probabilities = scipy.special.softmax(logits_all_final_task1, axis=1)
    # 对于二分类任务，计算有毒（类别1和2）的概率总和
    toxic_probabilities = probabilities[:, 1] + probabilities[:, 2]

    # 生成子群掩码
    num_subgroups = 6
    subgroup_masks = [(true_labels_task2 == i) for i in range(num_subgroups)]
    # print(subgroup_masks)

    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    for mask in subgroup_masks:
        # 计算Subgroup AUC
        # print(binary_labels.shape, toxic_probabilities.shape, mask.shape)
        subgroup_auc = calculate_auc(binary_labels, toxic_probabilities, mask)
        subgroup_aucs.append(subgroup_auc)

        # 计算BPSN AUC
        bpsn_mask = (mask & (binary_labels == 0)) | (~mask & (binary_labels == 1))
        bpsn_auc = calculate_auc(binary_labels, toxic_probabilities, bpsn_mask)
        bpsn_aucs.append(bpsn_auc)

        # 计算BNSP AUC
        bnsp_mask = (mask & (binary_labels == 1)) | (~mask & (binary_labels == 0))
        bnsp_auc = calculate_auc(binary_labels, toxic_probabilities, bnsp_mask)
        bnsp_aucs.append(bnsp_auc)

    print("subgroup_aucs", subgroup_aucs)
    print("bpsn_aucs", bpsn_aucs)
    print("bnsp_aucs", bnsp_aucs)
    # 计算GMB AUC
    gmb_subgroup_auc = calculate_gmb_auc(subgroup_aucs)
    gmb_bpsn_auc = calculate_gmb_auc(bpsn_aucs)
    gmb_bnsp_auc = calculate_gmb_auc(bnsp_aucs)

    print(f"GMB-Subgroup-AUC: {gmb_subgroup_auc}")
    print(f"GMB-BPSN-AUC: {gmb_bpsn_auc}")
    print(f"GMB-BNSP-AUC: {gmb_bnsp_auc}")
