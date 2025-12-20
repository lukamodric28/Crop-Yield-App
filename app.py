import streamlit as st
import pandas as pd
import joblib
import time
from function_transformers.add_continent_column import add_continent_column
from function_transformers.clip_year import clip_year
from function_transformers.log_transform import log_transform
from function_transformers.log_transform_without_poly_features import log_transform_without_poly_features
from function_transformers.output_engineered_features import output_engineered_features
from function_transformers.polynomial_features import polynomial_features

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌽", layout="wide", initial_sidebar_state="auto")

def show_predictor_page():
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
    st.markdown("<p style='font-size:18px; text-align:center; color:#FFFFFF;'>Predict the yield of your crops based on various factors using 6 models that you can choose from*! For input values, if you are unsure about the common/possible ranges, just hover over the question mark.</p>", unsafe_allow_html=True)

    st.divider()
    average_rainfall_in_mm_per_year = st.number_input("🌧️Enter the average annual rainfall(in mm)!: ", help = "Common ranges are typically 51mm - 3240mm.")
    avg_temp = st.number_input("🌡️Enter the average temperature(in °C)!: ", help = "Common ranges are typically 1°C - 30°C.")
    pesticides_in_tons_used = st.number_input("🧪Enter the amount of pesticides used(in tons)!: ", help = "Common ranges are typically 10 tons - 367778 tons.")
    year = st.number_input("📅Enter the year for which you want to predict the yield!: ", step = 1, format = "%i", help = "This model works best for years between 1990-2013, but it does work for years before or after that.")
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
                time.sleep(1)
                lasso_regression_prediction = loaded_final_lasso_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", lasso_regression_prediction[0].round(2))
        elif(Model == "Polynomial Regression"):
            with st.spinner("Predicting...Please wait!"):
                time.sleep(1)
                polynomial_regression_prediction = loaded_final_polynomial_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", polynomial_regression_prediction[0].round(2))
        elif(Model == "Random Forest Regression"):
            with st.spinner("Predicting...Please wait!"):
                time.sleep(1)
                random_forest_regression_prediction = loaded_final_random_forest_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", random_forest_regression_prediction[0].round(2))
        elif(Model == "Gradient Boosting"):
            with st.spinner("Predicting...Please wait!"):
                time.sleep(1)
                gradient_boosting_regression_prediction = loaded_final_gradient_boosting_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", gradient_boosting_regression_prediction[0].round(2))
        elif(Model == "K-Nearest Neighbors"):
            with st.spinner("Predicting...Please wait!"):
                time.sleep(1)
                k_nearest_neighbors_prediction = loaded_final_k_nearest_neighbors_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", k_nearest_neighbors_prediction[0].round(2))
        elif(Model == "Support Vector Regression"):
            with st.spinner("Predicting...Please wait!"):
                time.sleep(1)
                support_vector_regression_prediction = loaded_final_support_vector_regression_model.predict(user_input_df)
            st.write("Predicted Crop Yield(in Hectograms/Hectare): ", support_vector_regression_prediction[0].round(2))

    for i in range(0, 3):
        st.space()
    st.write("**These models are not perfect and may not always be accurate for all values given. It is recommended to test custom values given with the top 4 models to see the most accurate prediction and any variance. Another option would be to use values from the testing set to see how accurate the models are for those specific values.*")

def show_statistics_page():
    @st.cache_data
    def load_model_evaluation_data():
        return pd.read_csv("models_and_datasets/test_set_model_metrics.csv", index_col=0)
    
    st.title("Statistics During Model Testing And Evaluation📈📊")
    st.markdown("<p style='font-size:18px; text-align:center; color:#FFFFFF;'>Here are the statistics of various models during their testing and evaluation phase! This is to give you some insight on which models are highly accurate and which ones are not so accurate.</p>", unsafe_allow_html=True)
    st.divider()

    model_evaluation_data = load_model_evaluation_data()
    st.markdown("## Model Evaluation Metrics")
    st.dataframe(model_evaluation_data.style.format("{:.2f}"))

    url = "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset"
    st.write("The [full dataset](%s), which consisted of 28,242 data points, was split into a training set and a testing set using an 80-20 split. While splitting, stratification was used to better represent all the data. After the data was heavily trained, the models were tested on the testing set. These 4 metrics were used to evaluate the accuracies of the models. Down below, these metrics are described in detail." %url) 
    st.markdown("""
        <div style="padding-left: 40px; font-size: 15.5px;">
            <ul>
                <li style = "margin-bottom: 12px;">
                    <b><u>RMSE (Root Mean Squared Error)</u></b>: RMSE is a metric that measures the average magnitude of errors between the predicted and actual values and is very sensitive to outliers. It's calculated by squaring the differences between predicted and actual values, averaging them, and then taking the square root. A lower RMSE indicates better model performance, as it shows that the predictions are closer to the actual values.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>MSE (Mean Squared Error)</u></b>: MSE is a metric that is very similar to RMSE as it also measures the average magnitude of errors between predicted and actual values. However, instead of taking the square root of the average squared differences, MSE simply averages the squared differences. Like RMSE, a lower MSE indicates the model is doing better, as it shows that the predictions are closer to the actual values.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>MAE (Mean Absolute Error)</u></b>: MAE is a metric that measures the average absolute errors by treating all errors equally, making it robust to outliers. It's calculated very similarly to RMSE and MSE, but instead of squaring the average differences, MAE just takes the absolute value. A lower MAE indicates better model performance, as it shows that the predictions and the actual values are closer together.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>R²</u></b>: R^2 is a metric that is used to show how well a model fits the data based on the independent variable or variables given. It's measured by dividing the sum of the squared differences between the actual value and the dataset's mean value by the sum of the squared differences between the actual and predicted values. R^2 ranges from 0 to 1, with 1 being a perfect fit and 0 indicating no relationship. A higher R^2 value indicates that the model is better at predicting the data points and making relationships.
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## Rankings")
    st.write("Based on the above metrics(primarily RMSE), here are the rankings of the models from best to worst:")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥇K-Nearest Neighbors(KNN)")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥈Random Forest Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥉Gradient Boosting Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🏅Support Vector Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎖️Polynomial Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎖️Lasso Regression")

#def show_datasets_page():


pages = st.navigation([
    st.Page(show_predictor_page, title="Crop Yield Predictor", icon="🌽"),
    st.Page(show_statistics_page, title="Statistics During Model Testing And Evaluation", icon="📈")
])
pages.run() 