import numpy as np

# import some classification models
from sklearn.ensemble import RandomForestClassifier

from preprocessing import get_processed_data


if __name__=="__main__":
    train_data = get_processed_data(train=True, clean=True)
    test_data = get_processed_data(train=False, clean=True)

    # Partioning the data
    x_train = train_data.drop('income50K', axis=1)
    y_train = train_data['income50K']

    num_train = y_train.shape[0]

    x_test = test_data

    num_test = x_test.shape[0]

    model= RandomForestClassifier()
    model = model.fit(x_train, y_train) 

    y_pred = model.predict(x_test)

    y_pred = model.predict_proba(x_test)
    y_pred = np.add((-1)*y_pred[:, 0], y_pred[:, 1])

    result = [('ID','Prediction')]
    result.extend([(i, f"{pred:.2f}") for i, pred in enumerate(y_pred, 1)])

    with open("output.csv" , "w") as f:
        for x, y in result:
            f.write(f"{x},{y}\n")

