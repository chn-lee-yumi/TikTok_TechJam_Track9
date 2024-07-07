import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

labels = ["normal", "offensive", "hatespeech"]
target_groups = ["Race", "Religion", "Gender", "Sexual Orientation", "Miscellaneous", "None"]

label_to_index = {label: idx for idx, label in enumerate(labels)}
group_to_index = {group: idx for idx, group in enumerate(target_groups)}

device = torch.device("cuda")  # cuda mps cpu


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


with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_dataset = HateDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

# Model initialization
num_labels_task1 = len(labels)
num_labels_task2 = len(target_groups)
model = BertForMultiTaskClassificationAndTagging(model_name, num_labels_task1, num_labels_task2).to(device)

# Testing
model.load_state_dict(torch.load('model_4.pth'))
model.eval()

for input_ids, attention_mask, label, target_group, rationales in test_dataloader:
    input_ids, attention_mask, label, target_group, rationales = \
        input_ids.to(device), attention_mask.to(device), label.to(device), target_group.to(device), rationales.to(device)

    logits_task1, logits_task2, probs_tagging = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

    print(tokenizer.batch_decode(input_ids.tolist()))
    print(logits_task1)
    print(logits_task2)
    print(probs_tagging.view(3, 24))

    break
