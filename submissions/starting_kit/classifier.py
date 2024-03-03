from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

numerical_features = ['Age', 'Height', 'Weight', 'BMI', 'CO2', 'GDP', 'GNI',
                        'Life_expt','immunization', 'mort_rate', 'population',
                        'Number of athletes', 'Year']


class Classifier(BaseEstimator):
    def __init__(self):
        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, numerical_features),
            ], remainder='passthrough')  # We pass categorical data through

        # Initialize the CatBoostClassifier with some basic parameters
        self.model = CatBoostClassifier(iterations=100, learning_rate=0.1,
                                        depth=5, loss_function='MultiClass',
                                        verbose=0)

        self.pipe = make_imb_pipeline(SMOTE(random_state=42), self.model)

    def fit(self, X, y):
        self.pipe.fit(X, y)

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)