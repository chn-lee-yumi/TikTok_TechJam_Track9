import json
import os
import pickle

import more_itertools as mit


# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each == 1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each) == int:
            start = each
            end = each + 1
        elif len(each) == 2:
            start = each[0]
            end = each[1] + 1
        else:
            print('error')

        output.append({"docid": post_id,
                       "end_sentence": -1,
                       "end_token": end,
                       "start_sentence": -1,
                       "start_token": start,
                       "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    final_output = []

    if save_split:
        train_fp = open(save_path + 'train.jsonl', 'w')
        val_fp = open(save_path + 'val.jsonl', 'w')
        test_fp = open(save_path + 'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):

        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if majority_label == 'normal':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path + 'docs'):
                os.makedirs(save_path + 'docs')

            with open(save_path + 'docs/' + post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp) + '\n')
            else:
                print(post_id)

    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    return final_output


# The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
with open('post_id_divisions.json') as fp:
    id_division = json.load(fp)

with open(f'data_for_explainability.pkl', 'rb') as f:
    training_data = pickle.load(f)

method = 'union'
save_split = True
save_path = './explainability_data/'  # The dataset in Eraser Format will be stored here.
convert_to_eraser_format(training_data, method, save_split, save_path, id_division)


# 测试步骤：
# 执行 preprocess_for_explainability.py
# 执行 explainability.py
# 执行 explainability_get_result_json.py
# git clone git@github.com:jayded/eraserbenchmark.git
# 在 eraserbenchmark/rationale_benchmark/metrics.py 第285行后面添加    labels +=['normal']  # 这行是额外添加的！！！！！
# cd eraserbenchmark
# PYTHONPATH=./:$PYTHONPATH python3 rationale_benchmark/metrics.py --split test --strict --data_dir ../explainability_data --results ../explainability_data/results.json --score_file ../explainability_data/score.json
# 执行 print_explainability_result.py
