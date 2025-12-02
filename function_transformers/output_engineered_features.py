from sklearn.preprocessing import FunctionTransformer
def output_engineered_features(df):
  df = df.copy()
  df["rainfall_temperature"] = df["average_rainfall_in_mm_per_year"] * df["avg_temp"]
  df["pesticides_temperature"] = df["pesticides_in_tons_used"] * df["avg_temp"]
  df["rainfall_pesticides"] = df["average_rainfall_in_mm_per_year"] * df["pesticides_in_tons_used"]
  return df
feature_engineering_transformer = FunctionTransformer(output_engineered_features)