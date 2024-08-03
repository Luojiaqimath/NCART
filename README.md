Code for our paper: [NCART: Neural Classification and Regression Tree for Tabular Data](https://www.sciencedirect.com/science/article/pii/S0031320324003297?casa_token=yzENcV0nQOAAAAAA:r1sIyL0eMibEhu2q7tIha40YqXCtRy0BcN_NBgLDUADlwR0suHQa1YrOwURJFV-_-xlqy-kZ)


The supplementary materials are available in this repo.


See NCART_EXP for the datasets and the code.


## Binary classification
```python

import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ncart import NCARClassifier


data = load_breast_cancer()  
X = data.data.astype(np.float32)
y = data.target
feature_names = data.feature_names


# model = NCARClassifier(epochs=100, n_trees=8, n_layers=2, n_selected=6, use_gpu=False)  # CPU
# model = NCARClassifier(epochs=100, n_trees=8, n_layers=2, n_selected=6)  # single GPU
model = NCARClassifier(epochs=100, n_trees=8, n_layers=2, n_selected=6, data_parallel=True, gpu_ids=[0, 1])  # multiple GPU
model.fit(X, y)

importance = model.get_importance()

# Create a DataFrame with feature names and importance scores
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances using Seaborn
plt.figure()
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, width=0.6)
plt.title('Feature Importances', fontsize=12)
plt.xlabel('Importance Score', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show()

```

