import pickle

with open(f'test_data.pkl', 'rb') as f:
# with open(f'data_for_explainability.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
