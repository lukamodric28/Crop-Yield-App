from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn import set_config
log_transform_cols_without_poly_cols = ["pesticides_temperature", "rainfall_temperature", "rainfall_pesticides", "pesticides_in_tons_used"]
def log_transform_without_poly_features(df):
  df = df.copy()
  set_config(transform_output = "pandas")
  for i in log_transform_cols_without_poly_cols:
    df[i] = np.log1p(df[i])
  return df
log_columns_without_poly_features_transformer = FunctionTransformer(log_transform_without_poly_features)