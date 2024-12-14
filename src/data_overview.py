import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from const import TRAIN_DATA

cens = pd.read_csv(TRAIN_DATA, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', \
                                      'marital_status', 'occupation', 'relationship', 'race', 'sex', \
                                      'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income>50K'],
                                      skiprows=1)


print(cens.head())

print(cens.info())

# Total number of records
n_records = cens.shape[0]

# Total number of features
n_features = cens.shape[1]

# Number of records where individual's income is more than $50,000
n_greater_50k = cens[cens['income>50K'] == 1].shape[0]

# Number of records where individual's income is at most $50,000
n_at_most_50k = cens[cens['income>50K'] == 0].shape[0]

# Percentage of individuals whose income is more than $50,000
greater_percent =  (n_greater_50k / n_records) * 100

# Print the results
print("Total number of records: {}".format(n_records))
print("Total number of features: {}".format(n_features))
print("Individuals making more than $50k: {}".format(n_greater_50k))
print("Individuals making at most $50k: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50k: {:.2f}%".format(greater_percent))

# drop uneeded columns
cens.drop('education', inplace=True, axis=1)
cens.columns.tolist()

print("Looking for Null/invalid values")
# check for nulls
missing_values = cens.isnull().sum()
print(missing_values)


# check duplicates and remove it
print("Looking for duplicate rows that might deviate the data")
print("Before removing duplicates:", cens.duplicated().sum())

cens = cens[~cens.duplicated()]

print("After removing duplicates:", cens.duplicated().sum())

# before discarding
cens.sex.value_counts()

# discard spaces from entries
columns = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
for column in columns:
    cens[column] = cens[column].str.strip()

# after discarding
cens.sex.value_counts()

# before changing "?"
cens.workclass.value_counts()

# changing "?" to Unknown
change_columns = ['workclass', 'occupation', 'native_country']
for column in change_columns:
        cens[column] = cens[column].replace({'?': 'Unknown'})

# after changing "?"
cens.workclass.value_counts()

# a quick look on some statistics about the data
cens.describe()

# Heat map
plt.figure(figsize=[10,10])
 
ct_counts = cens.groupby(['education_num', 'income>50K']).size()
ct_counts = ct_counts.reset_index(name = 'count')
ct_counts = ct_counts.pivot(index = 'education_num', columns = 'income>50K', values = 'count').fillna(0)

sb.heatmap(ct_counts, annot = True, fmt = '.0f', cbar_kws = {'label' : 'Number of Individuals'})
plt.title('Number of People for Education Class relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Education Class')
plt.savefig("education.png")

# Clustered Bar Chart 
plt.figure(figsize=[8,6])
ax = sb.barplot(data = cens, x = 'income>50K', y = 'age', hue = 'sex')
ax.legend(loc = 8, ncol = 3, framealpha = 1, title = 'Sex')
plt.title('Average of Age for Sex relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Average of Age')
plt.savefig("age.png")

# Bar Chart 
plt.figure(figsize=[8,6])
sb.barplot(data=cens, x='income>50K', y='hours_per_week', palette='YlGnBu', hue="income>50K",  legend=False)
plt.title('Average of Hours per Week relative to Income')
plt.xlabel('Income ($)')
plt.ylabel('Average of Hours per Week')
plt.savefig("Working hours.png")