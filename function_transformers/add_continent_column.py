from sklearn.preprocessing import FunctionTransformer
from sklearn._config import set_config
import pycountry_convert as pc
import pandas as pd
def add_continent_column(df):
  df = df.copy()
  set_config(transform_output = "pandas")
  continents = []
  continent_dict = {
    "AS" : "Asia",
    "NA" : "North America",
    "SA" : "South America",
    "AF" : "Africa",
    "EU" : "Europe",
    "OC" : "Oceania"
}
  for i in df["Country"]:
    try:
      countries_alpha = pc.country_name_to_country_alpha2(i)
      country_continent_code = pc.country_alpha2_to_continent_code(countries_alpha)
      country_continent_name = continent_dict[country_continent_code]
      continents.append(country_continent_name)
    except Exception:
      print("There was an error")
      break
  continents_series = pd.Series(continents).to_numpy()
  df["Continent"] = continents_series
  return df
continent_column_transformer = FunctionTransformer(add_continent_column)