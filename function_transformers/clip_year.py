from sklearn.preprocessing import FunctionTransformer
def clip_year(df):
  df = df.copy()
  df["Year"].clip(lower = 1990, upper = 2013, inplace = True)
  return df
clip_year_transformer = FunctionTransformer(clip_year)