from sklearn import set_config
from sklearn.preprocessing import FunctionTransformer
import numpy as np
log_transform_cols = ["pesticides_temperature", "rainfall_temperature", "rainfall_pesticides", "pesticides_in_tons_used", "rainfall_temperature^2", "rainfall_temperature*rainfall_pesticides", "rainfall_pesticides^2"]
def log_transform(df):
  df = df.copy()
  set_config(transform_output = "pandas")
  for i in log_transform_cols:
    df[i] = np.log1p(df[i])
  return df
log_columns_transformer = FunctionTransformer(log_transform)