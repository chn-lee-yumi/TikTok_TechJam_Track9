import pickle
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# model_name = 'bert-large-uncased'
model_name = 'bert-base-uncased'
# model_name = 'bert-base-multilingual-uncased'
# model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two'
# model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
tokenizer = BertTokenizer.from_pretrained(model_name)

labels = ["normal", "offensive", "hatespeech"]
target_groups = ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous", "None"]

label_to_index = {label: idx for idx, label in enumerate(labels)}
group_to_index = {group: idx for idx, group in enumerate(target_groups)}

device = torch.device("cuda")  # cuda mps cpu
ratio = [1, 1, 100]  # loss ratio [task1, task2, probs_tagging]
# ratio = [1, 1, 1]  # loss ratio
accumulation_steps = 2
batch_size = 16


def set_seed(seed):
    # 设置 Python 随机数生成器的种子
    random.seed(seed)
    # 设置 NumPy 随机数生成器的种子
    np.random.seed(seed)
    # 设置 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    # 如果使用了 GPU，设置 CuDNN 随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置 CuDNN 后端为确定性算法
    torch.backends.cudnn.deterministic = True
    # 禁用 CuDNN 的自动优化
    torch.backends.cudnn.benchmark = False


# 调用函数并设置种子
set_seed(42)


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
            # nn.Softmax(dim=1),
        )  # labels
        self.classifier2 = nn.Sequential(
            nn.Linear(32, num_labels_task2),
            # nn.Softmax(dim=1),
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


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.bceloss = nn.BCELoss(reduction='none')  # 使用'reduction='none'来保留每个元素的损失值

    def forward(self, probs, targets, mask):
        loss = self.bceloss(probs, targets)
        masked_loss = loss * mask  # 只保留掩码标记为1的位置的损失
        return masked_loss.sum() / mask.sum()  # 归一化损失


# Load datasets
with open('train_data.pkl', 'rb') as f:
    # with open('test_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Initialize datasets and dataloaders
train_dataset = HateDataset(train_data)
val_dataset = HateDataset(val_data)
test_dataset = HateDataset(test_data)

sample_weights = calculate_weights(train_data)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Model initialization
num_labels_task1 = len(labels)
num_labels_task2 = len(target_groups)
model = BertForMultiTaskClassificationAndTagging(model_name, num_labels_task1, num_labels_task2).to(device)

# Loss functions and optimizer
criterion_task1 = nn.CrossEntropyLoss()
criterion_task2 = nn.CrossEntropyLoss()
# criterion_tagging = nn.BCELoss()
criterion_tagging = MaskedBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-8)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss_task1 = 0.0
    total_loss_task2 = 0.0
    total_loss_tagging = 0.0
    total_correct_task1 = 0
    total_correct_task2 = 0
    num = 0
    i = 0
    tqdm_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
    for input_ids, attention_mask, label, target_group, rationales in tqdm_dataloader:
        i += 1
        num += len(input_ids)
        input_ids, attention_mask, label, target_group, rationales = \
            input_ids.to(device), attention_mask.to(device), label.to(device), target_group.to(device), rationales.to(device)

        # Forward pass
        logits_task1, logits_task2, probs_tagging = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate losses
        loss_task1 = criterion_task1(logits_task1, torch.argmax(label, dim=1))
        loss_task2 = criterion_task2(logits_task2, torch.argmax(target_group, dim=1))
        # print(probs_tagging.shape, rationales.shape)  # torch.Size([32, 167, 1]) torch.Size([32, 167])
        # print(probs_tagging.shape, rationales.shape, attention_mask.shape)
        loss_tagging = criterion_tagging(probs_tagging.reshape(-1), rationales.view(-1), attention_mask.view(-1))

        total_loss_task1 += loss_task1.item()
        total_loss_task2 += loss_task2.item()
        total_loss_tagging += loss_tagging.item()

        # Calculate accuracy for task 1 and task 2
        total_correct_task1 += torch.sum(torch.argmax(logits_task1, dim=1) == torch.argmax(label, dim=1)).item()
        total_correct_task2 += torch.sum(torch.argmax(logits_task2, dim=1) == torch.argmax(target_group, dim=1)).item()

        # Backward pass and optimization
        loss = ratio[0] * loss_task1 + ratio[1] * loss_task2 + ratio[2] * loss_tagging
        # optimizer.zero_grad()
        loss = loss / accumulation_steps  # 平均损失
        loss.backward()
        # optimizer.step()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度

        # Update tqdm description with losses
        tqdm_dataloader.set_postfix(
            tagging_loss=total_loss_tagging / num,
            task1_loss=total_loss_task1 / num,
            task2_loss=total_loss_task2 / num,
        )

    # 如果最后的样本数不是accumulation_steps的倍数，进行一次额外的参数更新
    if len(train_dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Calculate average accuracy
    accuracy_task1 = total_correct_task1 / len(train_dataset)
    accuracy_task2 = total_correct_task2 / len(train_dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy - Task 1: {accuracy_task1:.4f}, Task 2: {accuracy_task2:.4f}, "
          f"Loss - Tagging: {total_loss_tagging / len(train_dataset):.4f}, "
          f"Task 1: {total_loss_task1 / len(train_dataset):.4f}, Task 2: {total_loss_task2 / len(train_dataset):.4f}"
          )

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss_task1 = 0.0
        total_loss_task2 = 0.0
        total_loss_tagging = 0.0
        total_correct_task1 = 0
        total_correct_task2 = 0
        for input_ids, attention_mask, label, target_group, rationales in val_dataloader:
            input_ids, attention_mask, label, target_group, rationales = \
                input_ids.to(device), attention_mask.to(device), label.to(device), target_group.to(device), rationales.to(device)

            logits_task1, logits_task2, probs_tagging = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate losses
            loss_task1 = criterion_task1(logits_task1, torch.argmax(label, dim=1))
            loss_task2 = criterion_task2(logits_task2, torch.argmax(target_group, dim=1))
            loss_tagging = criterion_tagging(probs_tagging.reshape(-1), rationales.view(-1), attention_mask.view(-1))

            total_loss_task1 += loss_task1.item()
            total_loss_task2 += loss_task2.item()
            total_loss_tagging += loss_tagging.item()

            # Calculate accuracy for task 1 and task 2
            total_correct_task1 += torch.sum(torch.argmax(logits_task1, dim=1) == torch.argmax(label, dim=1)).item()
            total_correct_task2 += torch.sum(torch.argmax(logits_task2, dim=1) == torch.argmax(target_group, dim=1)).item()

        # Calculate average accuracy
        accuracy_task1 = total_correct_task1 / len(val_dataset)
        accuracy_task2 = total_correct_task2 / len(val_dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Valid Accuracy - Task 1: {accuracy_task1:.4f}, Task 2: {accuracy_task2:.4f}, "
              f"Loss - Tagging: {total_loss_tagging / len(val_dataset):.4f}, "
              f"Task 1: {total_loss_task1 / len(val_dataset):.4f}, Task 2: {total_loss_task2 / len(val_dataset):.4f}"
              )

        torch.save(model.state_dict(), f'model_{epoch}.pth')

        # encoded_inputs = tokenizer(["fuck you, what are you doing? oh shit!",
        #                             "the weather is good!",
        #                             "A nigress too dumb to fuck has a scant chance of understanding anything beyond the size of a dick"
        #                             ], padding=True, truncation=True, return_tensors="pt")
        # logits_task1, logits_task2, probs_tagging = model(input_ids=encoded_inputs['input_ids'].to(device),
        #                                                   attention_mask=encoded_inputs['attention_mask'].to(device))
        # print(logits_task1)
        # print(logits_task2)
        # print(probs_tagging.view(3, 24))

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
        # loss_tagging = criterion_tagging(probs_tagging.reshape(-1), rationales.view(-1))
        loss_tagging = criterion_tagging(probs_tagging.reshape(-1), rationales.view(-1), attention_mask.view(-1))

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

    print(model_name, ratio)
    print("===== Report for Task 1 =====")
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
