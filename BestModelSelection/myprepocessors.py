
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# Extract first letter of variable cabin
class ExtractFirstLetter(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        return self

    def transform(self, X):
        X=X.copy()
        for var in self.variables:
            X[var] = X[var].str[0]
        return X


# Add missing indicator for numeric variables
class AddMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var + "_na"] = np.where(X[var].isnull(), 1, 0)
        return X


# Numerical missing value imputer
class NumericMissingImpute(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.numeric_dict_ = {}
        for var in self.variables:
            self.numeric_dict_[var] = X[var].median()
        # print(self.numeric_dict_)
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.numeric_dict_[var])
        return X


# Category missing value imputer
class CategoryMissingImpute(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna("Missing")
        return X


# Rare label encoding
class CategoryRareLabels(BaseEstimator, TransformerMixin):
    def __init__(self, variables, rare_perc=0.05):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.rare_perc = rare_perc

    def fit(self, X, y=None):
        self.rare_label_dict_ = {}
        for var in self.variables:
            tmp = X.groupby(var)[var].count() / len(X)
            self.rare_label_dict_[var] = tmp[tmp.values > self.rare_perc].index
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_label_dict_[var]), X[var], 'Rare')
        return X


# Category mapping in ascending
class CategoryMappings(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        tmp = X.copy()
        self.category_mappings_dict_ = {}
        for var in self.variables:
            ordered_labels = X.groupby([var])[var].count().sort_values().index
            self.category_mappings_dict_[var] = {k: i for i, k in enumerate(ordered_labels)}
            tmp[var] = tmp[var].map(self.category_mappings_dict_[var])
        self.features = list(pd.get_dummies(data=tmp, columns=self.variables, drop_first=True).columns)
        return self

    def transform(self, X):
        for var in self.variables:
            X[var] = X[var].map(self.category_mappings_dict_[var])
        X = pd.get_dummies(data=X, columns=self.variables, drop_first=True)
        for var in self.features:
            if var not in X.columns:
                X[var] = 0
        X = X[self.features]
        return X


class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)