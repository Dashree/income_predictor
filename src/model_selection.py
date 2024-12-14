import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# import some classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# import needed functions
from sklearn.model_selection import cross_validate

from preprocessing import get_processed_data



train_data = get_processed_data(train=True)
test_data = get_processed_data(train=False)

# Partioning the data
x_train = train_data.drop('income50K', axis=1)
y_train = train_data['income50K']

num_train = y_train.shape[0]

x_test = test_data

num_test = x_test.shape[0]

models = {}

# models with default parameter
models['GaussianNB'] = GaussianNB()
models['RandomForest'] = RandomForestClassifier()
models['AdaBoost'] = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=1), n_estimators=10, algorithm="SAMME")


max_acc = 0
max_acc_model_name = None

# Cross validation
for model_name in models:
    model = models[model_name]
    results = cross_validate(model, x_train, y_train, cv=5, scoring=['accuracy', 'roc_auc', 'f1_macro'], return_train_score=True)
    
    print(model_name + ":")
    print("Accuracy:" , 'train: ', results['train_accuracy'].mean(), '| test: ', results['test_accuracy'].mean())
    print("roc_auc-score:" , 'train: ', results['train_roc_auc'].mean(), '| test: ', results['test_roc_auc'].mean())
    print("F1 Score:" , 'train: ', results['train_f1_macro'].mean(), '| test: ', results['test_f1_macro'].mean())
    print("---------------------------------------------------------")

    if max_acc < results['test_roc_auc'].mean():
        max_acc_model_name = model_name


# View a list of the features and their importance scores
print('\nFeatures Importance:')
max_acc_model = models[max_acc_model_name]
max_acc_model = max_acc_model.fit(x_train, y_train) 
feat_imp = pd.DataFrame(zip(x_train.columns.tolist(), max_acc_model.feature_importances_ * 100), columns=['feature', 'importance'])
print(feat_imp)

# Features importance plot
plt.figure(figsize=[20,6])
sb.barplot(data=feat_imp, x='feature', y='importance')
plt.title('Features Importance', weight='bold', fontsize=20)
plt.xlabel('Feature', weight='bold', fontsize=13)
plt.ylabel('Importance (%)', weight='bold', fontsize=13)


# add annotations
impo = feat_imp['importance']
locs, labels = plt.xticks()

for loc, label in zip(locs, labels):
    count = impo[loc]
    pct_string = '{:0.2f}%'.format(count)

    plt.text(loc, count-0.8, pct_string, ha = 'center', color = 'w', weight='bold')
plt.savefig("feature importance.png")

data_final = train_data.copy()
data_final.drop(['race', 'sex', "education", 'capital.loss', 'native.country', 'fnlwgt', 'workclass','occupation'], axis=1, inplace=True)

print(max_acc_model_name)

