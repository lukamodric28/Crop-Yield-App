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
    col1, col2 = st.columns(2)
    with col1:
        average_rainfall_in_mm_per_year = st.number_input("🌧️Enter the average annual rainfall(in mm)!: ", help = "Common ranges are typically 51mm - 3240mm.")
        avg_temp = st.number_input("🌡️Enter the average temperature(in °C)!: ", help = "Common ranges are typically 1°C - 30°C.")
        pesticides_in_tons_used = st.number_input("🧪Enter the amount of pesticides used(in tons)!: ", help = "Common ranges are typically 10 tons - 367778 tons.")
    with col2:
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
    st.write("**These models are not perfect and may not always be accurate for all values given. It is recommended to test custom values given with the top 4 models to see the most accurate prediction and any variance. Another option would be to use values from the testing set in the 'Datasets Used While Developing the Models🌱' page to see how accurate the models are for those specific values.*")

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
    transposed_model_evaluation_data = model_evaluation_data.T
    st.bar_chart(transposed_model_evaluation_data[["RMSE"]])

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
                    <b><u>R²</u></b>: R² is a metric that is used to show how well a model fits the data based on the independent variable or variables given. It's measured by dividing the sum of the squared differences between the actual value and the dataset's mean value by the sum of the squared differences between the actual and predicted values. R^2 ranges from 0 to 1, with 1 being a perfect fit and 0 indicating no relationship. A higher R^2 value indicates that the model is better at predicting the data points and making relationships.
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## About the Models/Algorithms")
    st.write("Here is a description of each models, and its strengths and weaknesses:")
    st.markdown("""
        <div style="padding-left: 40px; font-size: 15.5px;">
            <ul>
                <li style = "margin-bottom: 12px;">
                    <b><u>Lasso Regression</u></b>: Lasso regression is a type of modified linear regression that tries to linearly correlate the target variable(in this case, the crop yield) with the features(in this case, average rainfall, temperature, pesticides used, etc.) by also adding a penalty(L1 regularization) to the coefficients of the independent variables. This penalty term helps to reduce the complexity of the model by shrinking some of the coefficients to zero. This makes it good for high dimensional data, its ability to perform feature selection, and to prevent overfitting(this means the model doesn't generalize enough and fits too closely to the training dataset). However, it fails when given non-linear relationships between the features and the target variable(like in this project), becomes very unstable when predictors are very correlated, introduces bias, etc.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>Polynomial Regression</u></b>: Polynomial regression is an extension of linear regression(or other linear algorithms) that models non-linear relationships between the target variable and the features by adding polynomial terms(raising the features to powers). It combines linear algorithms(in this project, ridge regression using L2 penalty that doesn't fully bring coefficients to zero but still penalizes large coefficients) with powers(squared terms are used in this project). This makes it somewhat decent for non-linear relationships, since it boosts a linear algorithm's ability to do so, and very versatile and adaptable. However, it's prone to overfitting, especially when the polynomial degree is high, and it can be computationally expensive. 
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>Random Forest Regression</u></b>: Random forest regression is a type of ensemble learning algorithm that uses multiple separate decision trees to predict the target variable. It works by creating multiple decision trees from random subsets of the data and features(called bagging), and then averages the predictions of all the trees to make a final prediction. This makes it very robust and accurate, can handle non-linear relationships very efficiently, and can handle high-dimensional data well. However, it's prone to overfitting, especially when the number of trees is high, can be computationally expensive, and can't extrapolate outside of the dataset given.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>KNN (K-Nearest Neighbors)</u></b>: KNN is a type of instance-based machine learning algorithm that predicts the target variable by looking at the "k" closest neighbors in the dataset. It works by calculating the distance between the input data point and all the other data points(depends what the k value is, it's 4 in this project) using a distance metric(called Manhattan distance in this project), and then averaging their predictions. Because of that, it's very good for non-linear relationships, simple to implement, and very versatile and adaptable when given new and unseen data. However, it can overfit, especially when the k value is low, can be computationally expensive for large datasets, and can't handle high-dimensional data well.
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>Gradient Boosting Regression</u></b>: Gradient boosting regression is a type of ensemble learning algorithm similar to random forest regression, but instead of using multiple decision trees, it builds the trees sequentially by focusing on the errors made by the previous trees. It works by creating a base model, then adding more models that focus on correcting the errors made by the previous models, and finally averaging their predictions. This makes gradient boosting regression have a high accuracy with non-linear relationships, can handle high-dimensional data well, and is very flexible and versatile. However, it's computationally expensive and time-consuming, prone to overfitting if not properly tuned, and is sensitive to outliers. 
                </li>
                <li style = "margin-bottom: 12px;">
                    <b><u>SVR (Support Vector Regression)</u></b>: SVR is a type of machine learning algorithm that finds the best-fit function(hyperplane) within a certain margin of tolerance to predict the target variable while still trying to keep the model "flat". It works by mapping the input data into a higher-dimensional space using a kernel function and then finding the hyperplane that best fits the data while minimizing the error within the tolerance. This makes SVR very effective for high-dimensional data, can handle non-linear relationships well(if using the radial basis function kernel like in this project), and is robust to outliers. However, it's computationally expensive, sensitive to hyperparameters, and has limited interpretability and poor scalability.  
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## Rankings")
    st.write("Based on the above metrics(primarily RMSE), here are the rankings of the models from best to worst:")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥇K-Nearest Neighbors(KNN)")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥈Random Forest Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🥉Support Vector Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🏅Gradient Boosting Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎖️Polynomial Regression")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎖️Lasso Regression")

def show_datasets_page():
    @st.cache_data
    def load_data():
        return pd.read_csv("models_and_datasets/yield_df.csv", index_col=0)
    @st.cache_data
    def load_testing_data():
        return pd.read_csv("models_and_datasets/crop_yield_testing_set.csv", index_col=0)
    @st.cache_data
    def load_training_data():
        return pd.read_csv("models_and_datasets/crop_yield_training_set.csv", index_col=0)
    
    crop_yield = load_data()
    crop_yield.rename(columns = {"Area":"Country", "Item":"Crop", "pesticides_tonnes":"pesticides_in_tons_used", "average_rain_fall_mm_per_year":"average_rainfall_in_mm_per_year"}, inplace=True)
    crop_yield = crop_yield.reindex(columns = ["Country", "Crop", "Year", "average_rainfall_in_mm_per_year", "pesticides_in_tons_used", "avg_temp", "hg/ha_yield"])
    st.title("Datasets Used While Developing the Models🌱")
    st.markdown("<p style='font-size:18px; text-align:center; color:#FFFFFF;'>Here is the full dataset, training set, and testing set used during the development of the models. To ensure that the strongest models are performing well, you can test values from the testing set.</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("## Full Dataset")
    st.write("This is the full dataset before stratification, consisting of 28,242 data points from Kaggle.")
    st.dataframe(crop_yield)

    crop_yield_testing_set = load_testing_data()
    st.markdown("## Testing Set")
    st.write("This is the testing set after stratification, consisting of 5,649 data points(20% of the full dataset) from the full dataset. This set is used to test the models and see how accurate they are when given unseen data.")
    st.dataframe(crop_yield_testing_set)

    crop_yield_training_set = load_training_data()
    st.markdown("## Training Set")
    st.write("This is the training set after stratification, consisting of 22,593 data points(80% of the full dataset) from the full dataset. This set is used to train the models and make them as accurate as possible before being feeding them to the testing set.")
    st.dataframe(crop_yield_training_set)

pages = st.navigation([
    st.Page(show_predictor_page, title="Crop Yield Predictor", icon="🌽"),
    st.Page(show_statistics_page, title="Statistics During Model Testing And Evaluation", icon="📈"),
    st.Page(show_datasets_page, title="Datasets Used While Developing the Models", icon="🌱")
])
pages.run() 