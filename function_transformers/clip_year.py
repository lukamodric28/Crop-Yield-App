from sklearn.preprocessing import FunctionTransformer
from sklearn._config import set_config
def clip_year(df):
  set_config(transform_output = "pandas")
  df = df.copy()
  df["Year"].clip(lower = 1990, upper = 2013, inplace = True)
  return df
clip_year_transformer = FunctionTransformer(clip_year)