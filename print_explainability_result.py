import json

with open('explainability_data/score.json') as fp:
    output_data = json.load(fp)

print('Plausibility')
print('IOU F1 :', output_data['iou_scores'][0]['macro']['f1'])
print('Token F1 :', output_data['token_prf']['instance_macro']['f1'])
print('AUPRC :', output_data['token_soft_metrics']['auprc'])
print()
print('Faithfulness')
print('Comprehensiveness :', output_data['classification_scores']['comprehensiveness'])
print('Sufficiency', output_data['classification_scores']['sufficiency'])
