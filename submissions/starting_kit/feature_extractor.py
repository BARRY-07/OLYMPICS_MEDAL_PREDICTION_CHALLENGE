class FeatureExtractor:
    def fit(self, X_df, y_array):
        """Fit method, which just returns the instance."""
        return self

    def transform(self, X_df):
        """Transform method, which just returns the input DataFrame."""
        return X_df
