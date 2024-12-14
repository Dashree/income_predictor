import pandas as pd
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from const import TRAIN_DATA, TEST_DATA


def get_processed_data(train=True, clean=False):
    
    data_loc = TEST_DATA
    data_col = {'age' : int, 'fnlwgt' : int, 'education.num': int, \
                                        'capital.gain' : int, 'capital.loss' : int, 'hours.per.week': int}
    if train:
        data_loc = TRAIN_DATA
        data_col['income>50K'] = int

    data = pd.read_csv(data_loc, header=0, encoding="ascii", skipinitialspace=True, sep=",", 
                       converters=data_col)
    
    if train is False:
        data.drop(["ID"], axis=1, inplace=True)

    # drop uneeded columns
    # check for nulls
    data.isna().sum()

    if train:
        data['income50K'] = data['income50K'].replace({0: -1})
    
    data = data[~data.duplicated()]

    # discard spaces from entries
    columns = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for column in columns:
        data[column] = data[column].str.strip()

    # changing "?" to Unknown
    change_columns = ['workclass', 'occupation', 'native.country']
    for column in change_columns:
            data[column] = data[column].replace({'?': 'Unknown'})

    numerical = ['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'fnlwgt']

    scaler = MinMaxScaler()
    data[numerical] = scaler.fit_transform(data[numerical])

    for col in data.columns:
        if data[col].dtypes == 'object':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    if clean:
        data.drop(['race', 'sex', "education", 'capital.loss', 'native.country', 'fnlwgt', 'workclass','occupation'], axis=1, inplace=True)

    return data