import streamlit as st
import pandas as pd
import joblib
from function_transformers.add_continent_column import add_continent_column
from function_transformers.clip_year import clip_year
from function_transformers.log_transform import log_transform
from function_transformers.log_transform_without_poly_features import log_transform_without_poly_features
from function_transformers.output_engineered_features import output_engineered_features
from function_transformers.polynomial_features import polynomial_features

page_selector = st.sidebar.selectbox("Select the page you want to go to!", ["Crop Yield Predictor🌽", "Statistics During Model Evaluation📈"])

if(page_selector == "Crop Yield Predictor🌽"):
    @st.cache_data
    def load_data():
        return pd.read_csv("models_and_datasets/yield_df.csv")

    crop_yield = load_data()
    countries = sorted(crop_yield["Area"].dropna().unique())
    crop = sorted(crop_yield["Item"].dropna().unique())
    loaded_final_lasso_regression_model = joblib.load("models_and_datasets/lasso_regression_best_model.joblib")
    loaded_final_polynomial_regression_model = joblib.load("models_and_datasets/polynomial_regression_best_model.joblib")
    loaded_final_random_forest_regression_model = joblib.load("models_and_datasets/random_forest_regression_best_model.joblib")
    loaded_final_gradient_boosting_regression_model = joblib.load("models_and_datasets/gradient_boosting_best_model.joblib")
    loaded_final_k_nearest_neighbors_model = joblib.load("models_and_datasets/k_nearest_neighbors_best_model.joblib")
    loaded_final_support_vector_regression_model = joblib.load("models_and_datasets/support_vector_regression_best_model.joblib")

    st.title("Crop Yield Predictor🌽🌾🍚")
    st.markdown("<p style='font-size:20px; text-align:center; color:#FFFFFF;'>Predict the yield of your crops based on various factors using 6 models that you can choose from! For input values, if you are unsure about the common/possible ranges, just hover over the question mark.</p>", unsafe_allow_html=True)

    st.divider()
    average_rainfall_in_mm_per_year = st.number_input("🌧️Enter the average annual rainfall(in mm)!: ", help = "Common ranges are typically 51mm - 3240mm")
    avg_temp = st.number_input("🌡️Enter the average temperature(in °C)!: ", help = "Common ranges are typically 1°C - 30°C")
    pesticides_in_tons_used = st.number_input("🧪Enter the amount of pesticides used(in tons)!: ", help = "Common ranges are typically 10 tons - 367778 tons")
    year = st.number_input("📅Enter the year for which you want to predict the yield!: ", help = "This model works best for years between 1990-2013, but it does work for years before or after that")
    Country = st.selectbox("Select the country you want to predict the yield for!", countries)
    Crop = st.selectbox("Select the crop you want to predict the yield for!", crop)
    Model = st.selectbox("Select the model you want to use for prediction!", ["Lasso Regression", "Polynomial Regression", "Random Forest Regression", "Gradient Boosting", "K-Nearest Neighbors", "Support Vector Regression"], help = "To know more about which model to choose, go to the 'Statistics During Model Evaluation📈' page from the sidebar.")

    user_input_df = pd.DataFrame({
        "average_rainfall_in_mm_per_year": [average_rainfall_in_mm_per_year],
        "avg_temp": [avg_temp],
        "pesticides_in_tons_used": [pesticides_in_tons_used],
        "Year": [year],
        "Country": [Country],
        "Crop": [Crop]
    })

    if(st.button("Predict Yield")):
        if(Model == "Lasso Regression"):
            with st.spinner("Predicting...Please wait!"):
                lasso_regression_prediction = loaded_final_lasso_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", lasso_regression_prediction[0])
        elif(Model == "Polynomial Regression"):
            with st.spinner("Predicting...Please wait!"):
                polynomial_regression_prediction = loaded_final_polynomial_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", polynomial_regression_prediction[0])
        elif(Model == "Random Forest Regression"):
            with st.spinner("Predicting...Please wait!"):
                random_forest_regression_prediction = loaded_final_random_forest_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", random_forest_regression_prediction[0])
        elif(Model == "Gradient Boosting"):
            with st.spinner("Predicting...Please wait!"):
                gradient_boosting_regression_prediction = loaded_final_gradient_boosting_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", gradient_boosting_regression_prediction[0])
        elif(Model == "K-Nearest Neighbors"):
            with st.spinner("Predicting...Please wait!"):
                k_nearest_neighbors_prediction = loaded_final_k_nearest_neighbors_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", k_nearest_neighbors_prediction[0])
        elif(Model == "Support Vector Regression"):
            with st.spinner("Predicting...Please wait!"):
                support_vector_regression_prediction = loaded_final_support_vector_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield: ", support_vector_regression_prediction[0])

if(page_selector == "Statistics During Model Evaluation📈"):
    st.title("Statistics During Model Testing And Evaluation📈📊")
    st.markdown("<p style='font-size:20px; text-align:center; color:#FFFFFF;'>Here are the statistics of various models during their testing and evaluation phase! This is to give you some insight on which models are highly accurate and which ones are not so accurate.</p>", unsafe_allow_html=True)
    st.divider()
    @st.cache_data
    def load_model_evaluation_data():
        return pd.read_csv("models_and_datasets/test_set_model_metrics.csv", index_col=0)
    model_evaluation_data = load_model_evaluation_data()
    st.markdown("### Model Evaluation Metrics")
    st.dataframe(model_evaluation_data.style.format("{:.2f}"))

    