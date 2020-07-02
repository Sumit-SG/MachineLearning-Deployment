import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import myconfig as config
import mypipeline as pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import joblib

param_grid = [{'classifier': [LogisticRegression()],
               'classifier__penalty': ['l1', 'l2'],
               'classifier__C': [1.0, 0.5, 0.1],
               'classifier__solver': ['liblinear']},

              {'classifier': [RandomForestClassifier()],
               'classifier__n_estimators': list(range(10, 101, 10)),
               'classifier__max_features': list([2, 4, 6, 8]),
               'classifier__random_state': [0]}
              ]


def run_training():
    data = pd.read_csv(config.DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)
    # print(X_train.shape)
    # print()

    gscv = GridSearchCV(pipeline.titanic_pipe, param_grid, cv=2, n_jobs=2, return_train_score=False, verbose=1)
    best_model = gscv.fit(X_train, y_train)
    # gscv.fit(X_train, y_train)
    # print(best_model.best_estimator_['classifier'])
    # print(best_model.predict(X_train))
    print(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
    # print(pipeline.titanic_pipe)
    joblib.dump(gscv, config.PIPELINE_PATH)

    # pipeline.titanic_pipe.fit(X_train, y_train)
    # joblib.dump(pipeline.titanic_pipe,config.PIPELINE_PATH)


if __name__ == "__main__":
    run_training()
    print("training completed")
