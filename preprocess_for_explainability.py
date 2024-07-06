import json
import pickle
from collections import Counter

import numpy as np
from transformers import BertTokenizer

# 定义类别到目标群体的映射表
category_to_target_group = {
    "African": "Race",
    "Arab": "Race",
    "Asian": "Race",
    "Indian": "Race",
    "Caucasian": "Race",
    "Hispanic": "Race",
    "Buddhism": "Religion",
    "Christian": "Religion",
    "Hindu": "Religion",
    "Islam": "Religion",
    "Jewish": "Religion",
    "Nonreligious": "Religion",
    "Men": "Gender",
    "Women": "Gender",
    "Heterosexual": "Sexual Orientation",
    "Homosexual": "Sexual Orientation",
    "Bisexual": "Sexual Orientation",
    "Asexual": "Sexual Orientation",
    "Gay": "Sexual Orientation",
    "Indigenous": "Miscellaneous",
    "Immigrant": "Miscellaneous",
    "Refugee": "Miscellaneous",
    "Disability": "Miscellaneous",
    "Economic": "Miscellaneous",
    "Minority": "Miscellaneous",
    "Other": "Miscellaneous",
    "None": "None",
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 167  # base on the dataset, this is the longest input


def handle_tokens(original_tokens, rationales=None):
    if not rationales:
        rationales = [0] * len(original_tokens)
    assert len(original_tokens) == len(rationales)
    new_tokens = []
    new_rationales = []

    for token, rationale in zip(original_tokens, rationales):
        sub_tokens = tokenizer.encode(token, add_special_tokens=False, padding=False)
        new_tokens.extend(sub_tokens)
        new_rationales.extend([rationale] * len(sub_tokens))
    # print(new_tokens, new_rationales)
    assert len(new_tokens) == len(new_rationales)
    return new_tokens, new_rationales


def preprocess_data(json_file, divisions_file):
    def majority_target(annotators, key='target'):
        targets = [target for annotator in annotators for target in annotator[key]]
        most_common = Counter(targets).most_common(1)
        return most_common[0][0] if most_common else "None"

    def majority_label(annotators):
        targets = [annotator["label"] for annotator in annotators]
        most_common = Counter(targets).most_common(1)
        return most_common[0][0] if most_common else "None"

    def calculate_softmax(rationales):
        rationales_sum = np.sum(rationales, axis=0)
        e_x = np.exp(rationales_sum - np.max(rationales_sum))
        return e_x / e_x.sum(axis=0)

    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    data = load_json(json_file)

    processed_data = []
    max_len = 0
    for post_id, post_data in data.items():
        # 对 target 进行投票并转换为目标群体
        target = category_to_target_group[majority_target(post_data['annotators'])]
        label = majority_label(post_data['annotators'])
        # if label != "normal" and target == "None":  # 存在的
        #     print(label, target)
        #     print(post_data['annotators'])

        tokens = post_data['post_tokens']

        # 计算 rationales 的 softmax
        if post_data['rationales']:
            valid_rationales = [r for r in post_data['rationales'] if len(r) == len(post_data['post_tokens'])]
            # if len(valid_rationales) != len(post_data['rationales']):
            #     print(f"Removed {len(post_data['rationales']) - len(valid_rationales)} invalid rationales.")
            rationales_softmax = calculate_softmax(valid_rationales).tolist()
            tokens, valid_rationales = handle_tokens(tokens, rationales_softmax)
            rationales_softmax = [0] + valid_rationales + [0]
        else:
            tokens, _ = handle_tokens(tokens)
            token_count = len(tokens)
            rationales_softmax = np.full(token_count, 1 / token_count).tolist()
            rationales_softmax = [0] + rationales_softmax + [0]
        tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        assert len(tokens) == len(rationales_softmax)
        max_len = max(max_len, len(tokens))
        # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))
        # attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in token_ids]
        # rationales_softmax = rationales_softmax + [0] * (max_length - len(rationales_softmax))

        processed_data.append([post_id, label, tokens, post_data['rationales'], [annotator["label"] for annotator in post_data['annotators']]])

    # 将处理后的数据保存为 pickle 文件
    with open(f'data_for_explainability.pkl', 'wb') as f:
        pickle.dump(processed_data, f)

    print("max_len", max_len)


# 示例用法
if __name__ == "__main__":
    json_file = 'dataset.json'
    divisions_file = 'post_id_divisions.json'
    preprocess_data(json_file, divisions_file)
