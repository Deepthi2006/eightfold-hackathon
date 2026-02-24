import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import agent
from agent import run_agent

# Load data
df = pd.read_csv('dataset.csv')
labels = np.load('labels.npy')

class MockOracle:
    def __init__(self, labels):
        self.labels = labels
        self.used = 0
    def __call__(self, indices):
        self.used += len(indices)
        return [int(self.labels[i]) for i in indices]

o = MockOracle(labels)
preds = run_agent(df, o, 100)

f1 = f1_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)

print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Queries:   {o.used}")
