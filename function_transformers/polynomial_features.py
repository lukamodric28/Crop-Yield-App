from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
def polynomial_features(df):
  df = df.copy()
  poly_features = PolynomialFeatures(degree=2, include_bias=False)
  base_cols = ["rainfall_temperature", "rainfall_pesticides"]

  poly_array = poly_features.fit_transform(df[base_cols])
  poly_cols = poly_features.get_feature_names_out(base_cols)

  wanted_cols = poly_cols[2:]
  df[wanted_cols] = poly_array.iloc[:, 2:]

  df.rename(columns={
      "rainfall_temperature rainfall_pesticides": "rainfall_temperature*rainfall_pesticides"
  }, inplace=True)
  return df
polynomial_features_transformer = FunctionTransformer(polynomial_features, validate = False)
