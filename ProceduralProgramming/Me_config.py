DATA_PATH = "titanic.csv"
SCALER_PATH = 'scaler_train.pkl'
MODEL_PATH =  'logistic_regression.pkl'

TARGET = 'survived'

NUMERIC_IMPUTE = ['age', 'fare']
CATEGORICAL_IMPUTE = ['cabin', 'embarked']


FREQUENCY_LABELS = {
    'sex': ['female', 'male'],
    'cabin': ['C', 'Missing'],
    'embarked': ['C', 'Q', 'S'],
    'title': ['Miss', 'Mr', 'Mrs']
}

ENCODING_CATEGORICAL = {
    'sex': {'female': 0, 'male': 1},
    'cabin': {'C': 0, 'Rare': 1, 'Missing': 2},
    'embarked': {'Rare': 0, 'Q': 1, 'C': 2, 'S': 3},
    'title': {'Rare': 0, 'Mrs': 1, 'Miss': 2, 'Mr': 3}
}

NUMERIC_MEDIAN = {
    'age': 28.0,
    'fare': 14.4542
}

CATEGORICAL_VARIABLES = ['sex', 'cabin', 'embarked', 'title']

VARIABLE_LIST = ['pclass', 'age', 'sibsp','parch','fare','age_na','fare_na','sex_1','cabin_1','cabin_2','embarked_1',
                 'embarked_2','embarked_3','title_1','title_2','title_3']


