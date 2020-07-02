from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import myprepocessors as pp
import myconfig as config

titanic_pipe = Pipeline(
    [
        ('extract_letter', pp.ExtractFirstLetter(variables=config.VARIABLE_LETTER)),
        ('missing_indicator', pp.AddMissingIndicator(variables=config.NUMERIC_IMPUTE)),
        ('numerical_imputer', pp.NumericMissingImpute(variables=config.NUMERIC_IMPUTE)),
        ('categorical_imputer', pp.CategoryMissingImpute(variables=config.CATEGORICAL_IMPUTE)),
        ('rare_label_encoding', pp.CategoryRareLabels(variables=config.CATEGORICAL_VARS)),
        ('category_mappings', pp.CategoryMappings(variables=config.CATEGORICAL_VARS)),
        ('scaler', StandardScaler()),
        ('classifier',pp.ClfSwitcher(estimator=LogisticRegression()))
    ]

)
#