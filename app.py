import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import time
from function_transformers.add_continent_column import add_continent_column
from function_transformers.clip_year import clip_year
from function_transformers.log_transform import log_transform
from function_transformers.log_transform_without_poly_features import log_transform_without_poly_features
from function_transformers.output_engineered_features import output_engineered_features
from function_transformers.polynomial_features import polynomial_features

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌽", layout="wide", initial_sidebar_state="auto")

loaded_final_lasso_regression_model = joblib.load("models_and_datasets/lasso_regression_best_model.joblib")
loaded_final_polynomial_regression_model = joblib.load("models_and_datasets/polynomial_regression_best_model.joblib")
loaded_final_random_forest_regression_model = joblib.load("models_and_datasets/random_forest_regression_best_model.joblib")
loaded_final_gradient_boosting_regression_model = joblib.load("models_and_datasets/gradient_boosting_best_model.joblib")
loaded_final_k_nearest_neighbors_model = joblib.load("models_and_datasets/k_nearest_neighbors_best_model.joblib")
loaded_final_support_vector_regression_model = joblib.load("models_and_datasets/support_vector_regression_best_model.joblib")

crop_benchmarks = {
    "Cassava": [95000, 130000],
    "Maize": [37000, 61000],
    "Potatoes": [170000, 215000],
    "Rice, paddy": [38000, 48000],
    "Plantains and others": [50000, 68000],
    "Sorghum": [13000, 16000],
    "Soybeans": [19000, 30000],
    "Sweet potatoes": [110000, 160000],
    "Wheat": [25000, 38000],
    "Yams": [80000, 110000]
}

annual_growth_rates = {
    "Cassava": 0.028,
    "Maize": 0.022,
    "Potatoes": 0.024,
    "Rice, paddy": 0.010,
    "Plantains and others": 0.008,
    "Sorghum": 0.010,
    "Soybeans": 0.016,
    "Sweet potatoes": 0.018,
    "Wheat": 0.014,
    "Yams": 0.018
}

if "average_rainfall_in_mm_per_year" not in st.session_state:
    st.session_state.average_rainfall_in_mm_per_year = 0
if "avg_temp" not in st.session_state:
    st.session_state.avg_temp = 0.00
if "pesticides_in_tons_used" not in st.session_state:
    st.session_state.pesticides_in_tons_used = 0.00
if "year" not in st.session_state:
    st.session_state.year = 1990
if "Country" not in st.session_state:
    st.session_state.Country = "Albania"
if "Crop" not in st.session_state:
    st.session_state.Crop = "Cassava"
if "Model" not in st.session_state:
    st.session_state.Model = "Lasso Regression"

models = ["Lasso Regression", "Polynomial Regression", "Random Forest Regression", "Gradient Boosting", "K-Nearest Neighbors", "Support Vector Regression"]

def show_predictor_page():
    @st.cache_data
    def load_data():
        return pd.read_csv("models_and_datasets/yield_df.csv")

    crop_yield = load_data()
    countries = sorted(crop_yield["Area"].dropna().unique())
    crop = sorted(crop_yield["Item"].dropna().unique())

    st.markdown("<h1 style = 'text-align:center; font-size:50px; color:#FFFFFF;'><b>Crop Yield Predictor🌽🌾🍚</b></h1>", unsafe_allow_html=True)
    st.markdown("<p style = 'font-size:18px; text-align:center; color:#FFFFFF; margin-top:-5px'>Predict the yield of your crops based on various factors using 6 models that you can choose from! For input values, if you are unsure about the common/possible ranges, just hover over the question mark.</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.average_rainfall_in_mm_per_year = st.number_input("🌧️Enter the average annual rainfall(in mm)!: ", help = "Common ranges are typically 51mm - 3240mm.", min_value = 0, value = st.session_state.average_rainfall_in_mm_per_year)
        st.session_state.avg_temp = st.number_input("🌡️Enter the average temperature(in °C)!: ", help = "Common ranges are typically 1°C - 30°C.", min_value = 0.00, value = st.session_state.avg_temp, format = "%.2f", step = 1.00)
        st.session_state.pesticides_in_tons_used = st.number_input("🧪Enter the amount of pesticides used(in tons)!: ", help = "Common ranges are typically 10 tons - 367778 tons.", min_value = 0.00, format = "%.2f", value = st.session_state.pesticides_in_tons_used, step = 1.00)
    with col2:
        st.session_state.year = st.number_input("📅Enter the year for which you want to predict the yield!: ", step = 1, format = "%i", help = "This model was trained for years 1990-2013. If years are outside this range, it will be clipped to either the maximum or minimum year and a uncertainty range will be added to prevet innaccurate predictions.", min_value = 0, value = st.session_state.year)
        st.session_state.Country = st.selectbox("📍Select the country you want to predict the yield for!", countries, index = countries.index(st.session_state.Country))
        st.session_state.Crop = st.selectbox("🌾Select the crop you want to predict the yield for!", crop, index = crop.index(st.session_state.Crop))
    st.session_state.Model = st.selectbox("🤖 Select the model you want to use for prediction!", models, index = models.index(st.session_state.Model), help = "To know more about which model to choose, go to the '📈Statistics' page from the sidebar.")

    user_input_df = pd.DataFrame({
        "average_rainfall_in_mm_per_year": [st.session_state.average_rainfall_in_mm_per_year],
        "avg_temp": [st.session_state.avg_temp],
        "pesticides_in_tons_used": [st.session_state.pesticides_in_tons_used],
        "Year": [st.session_state.year],
        "Country": [st.session_state.Country],
        "Crop": [st.session_state.Crop]
    })

    if(st.button("Predict Yield", type = "primary")):
        with st.spinner("Predicting...Please wait!"):
            uncertainty_value = 0
            rate = annual_growth_rates[st.session_state.Crop]
            benchmarks = crop_benchmarks.get(st.session_state.Crop)
            low, high = benchmarks[0], benchmarks[1]
            range_str = f"{low:,} - {high:,}"

            if(st.session_state.year > 2013):
                uncertainty_value = (st.session_state.year - 2013) * rate
            elif(st.session_state.year < 1990):
                uncertainty_value = (1990 - st.session_state.year) * rate

            if(st.session_state.Model == "Lasso Regression"):
                time.sleep(1)
                lasso_regression_prediction = loaded_final_lasso_regression_model.predict(user_input_df)
                lasso_regression_prediction_multiplied = lasso_regression_prediction[0] * uncertainty_value

                if lasso_regression_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {lasso_regression_prediction[0].round(2)} ± {round(lasso_regression_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", lasso_regression_prediction[0].round(2)) 

                if(lasso_regression_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= lasso_regression_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
            elif(st.session_state.Model == "Polynomial Regression"):
                time.sleep(1)
                polynomial_regression_prediction = loaded_final_polynomial_regression_model.predict(user_input_df)
                polynomial_regression_prediction_multiplied = polynomial_regression_prediction[0] * uncertainty_value

                if polynomial_regression_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {polynomial_regression_prediction[0].round(2)} ± {round(polynomial_regression_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", polynomial_regression_prediction[0].round(2))

                if(polynomial_regression_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= polynomial_regression_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
            elif(st.session_state.Model == "Random Forest Regression"):
                time.sleep(1)
                random_forest_regression_prediction = loaded_final_random_forest_regression_model.predict(user_input_df)
                random_forest_regression_prediction_multiplied = random_forest_regression_prediction[0] * uncertainty_value

                if random_forest_regression_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {random_forest_regression_prediction[0].round(2)} ± {round(random_forest_regression_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", random_forest_regression_prediction[0].round(2))

                if(random_forest_regression_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= random_forest_regression_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
            elif(st.session_state.Model == "Gradient Boosting"):
                time.sleep(1)
                gradient_boosting_regression_prediction = loaded_final_gradient_boosting_regression_model.predict(user_input_df)
                gradient_boosting_regression_prediction_multiplied = gradient_boosting_regression_prediction[0] * uncertainty_value

                if gradient_boosting_regression_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {gradient_boosting_regression_prediction[0].round(2)} ± {round(gradient_boosting_regression_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", gradient_boosting_regression_prediction[0].round(2))

                if(gradient_boosting_regression_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= gradient_boosting_regression_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
            elif(st.session_state.Model == "K-Nearest Neighbors"):
                time.sleep(1)
                k_nearest_neighbors_prediction = loaded_final_k_nearest_neighbors_model.predict(user_input_df)
                k_nearest_neighbors_prediction_multiplied = k_nearest_neighbors_prediction[0] * uncertainty_value

                if k_nearest_neighbors_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {k_nearest_neighbors_prediction[0].round(2)} ± {round(k_nearest_neighbors_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", k_nearest_neighbors_prediction[0].round(2))

                if(k_nearest_neighbors_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= k_nearest_neighbors_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
            elif(st.session_state.Model == "Support Vector Regression"):
                time.sleep(1)
                support_vector_regression_prediction = loaded_final_support_vector_regression_model.predict(user_input_df)
                support_vector_regression_prediction_multiplied = support_vector_regression_prediction[0] * uncertainty_value

                if support_vector_regression_prediction_multiplied > 0:
                    st.write(f"Predicted Crop Yield(in Hectograms/Hectare): {support_vector_regression_prediction[0].round(2)} ± {round(support_vector_regression_prediction_multiplied, 2)}")
                else:
                    st.write("Predicted Crop Yield(in Hectograms/Hectare): ", support_vector_regression_prediction[0].round(2))

                if(support_vector_regression_prediction[0].round(2) < low):
                    st.warning(f"This is below the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                elif(low <= support_vector_regression_prediction[0].round(2) <= high):
                    st.success(f"This is within the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")
                else:
                    st.success(f"This is above the average global crop yield for {st.session_state.Crop}, which was {range_str} hectograms per hectare.")

    st.info("💡**Tip**: These models are not perfect and may not always be accurate for all values given. It is recommended to test custom values given with the top 4 models to see the most accurate prediction and any variance. Another option would be to use values from the testing set in the '🌱Datasets' page to see how accurate the models are for those specific values.")

def show_statistics_page():
    @st.cache_data
    def load_model_evaluation_data():
        return pd.read_csv("models_and_datasets/test_set_model_metrics.csv", index_col=0)
    
    st.markdown("<h1 style = 'text-align:center; font-size:46px; color:#FFFFFF;'><b>Statistics During Model Testing And Evaluation📈📊</b></h1>", unsafe_allow_html=True)
    st.markdown("<p style = 'font-size:18px; text-align:center; color:#FFFFFF; margin-top:-5px'>Here are the statistics of various models during their testing and evaluation phase! This is to give you some insight on which models are highly accurate and which ones are not so accurate.</p>", unsafe_allow_html=True)
    st.divider()

    model_evaluation_data = load_model_evaluation_data()
    st.markdown("## Model Evaluation Metrics")
    st.dataframe(model_evaluation_data.style.format("{:.2f}"))

    url = "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset"
    st.info("The [full dataset](%s), which consisted of 28,242 data points, was split into a training set and a testing set using an 80-20 split. While splitting, stratification was used to better represent all the data. After the data was heavily trained, the models were tested on the testing set. These 4 metrics were used to evaluate the accuracies of the models. Down below, these metrics are described in detail." %url) 
    with st.expander("🔍 Detailed Descriptions of Evaluation Metrics", expanded=False):
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Rankings")
        st.markdown("Based on RMSE (Best to Worst):")
        st.markdown('<div class="rank-item">🥇 K-Nearest Neighbors (KNN)</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-item">🥈 Random Forest Regression</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-item">🥉 Support Vector Regression</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-item">🏅 Gradient Boosting Regression</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-item">🎖️ Polynomial Regression</div>', unsafe_allow_html=True)
        st.markdown('<div class="rank-item">🎖️ Lasso Regression</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("## About the Models/Algorithms")
        st.write("Here is a description of each models, and its strengths and weaknesses:")
        tab1, tab2, tab3 = st.tabs(["Linear/Polynomial Regression", "Random Forest/Gradient Boosting", "KNN/SVR"])

        with tab1:
            st.markdown("### Lasso Regression")
            st.write("Lasso regression is a type of modified linear regression that tries to linearly correlate the target variable(in this case, the crop yield) with the features(in this case, average rainfall, temperature, pesticides used, etc.) by also adding a penalty(L1 regularization) to the coefficients of the independent variables. This penalty term helps to reduce the complexity of the model by shrinking some of the coefficients to zero. This makes it good for high dimensional data, its ability to perform feature selection, and to prevent overfitting(this means the model doesn't generalize enough and fits too closely to the training dataset). However, it fails when given non-linear relationships between the features and the target variable(like in this project), becomes very unstable when predictors are very correlated, introduces bias, etc.")

            st.markdown("### Polynomial Regression")
            st.write("Polynomial regression is an extension of linear regression(or other linear algorithms) that models non-linear relationships between the target variable and the features by adding polynomial terms(raising the features to powers). It combines linear algorithms(in this project, ridge regression using L2 penalty that doesn't fully bring coefficients to zero but still penalizes large coefficients) with powers(squared terms are used in this project). This makes it somewhat decent for non-linear relationships, since it boosts a linear algorithm's ability to do so, and very versatile and adaptable. However, it's prone to overfitting, especially when the polynomial degree is high, and it can be computationally expensive.")

        with tab2:
            st.markdown("### Random Forest Regression")
            st.write("Random forest regression is a type of ensemble learning algorithm that uses multiple separate decision trees to predict the target variable. It works by creating multiple decision trees from random subsets of the data and features(called bagging), and then averages the predictions of all the trees to make a final prediction. This makes it very robust and accurate, can handle non-linear relationships very efficiently, and can handle high-dimensional data well. However, it's prone to overfitting, especially when the number of trees is high, can be computationally expensive, and can't extrapolate outside of the dataset given.")

            st.markdown("### Gradient Boosting Regression")
            st.write("Gradient boosting regression is a type of ensemble learning algorithm similar to random forest regression, but instead of using multiple decision trees, it builds the trees sequentially by focusing on the errors made by the previous trees. It works by creating a base model, then adding more models that focus on correcting the errors made by the previous models, and finally averaging their predictions. This makes gradient boosting regression have a high accuracy with non-linear relationships, can handle high-dimensional data well, and is very flexible and versatile. However, it's computationally expensive and time-consuming, prone to overfitting if not properly tuned, and is sensitive to outliers.")

        with tab3:
            st.markdown("### KNN (K-Nearest Neighbors)")
            st.write("KNN is a type of instance-based machine learning algorithm that predicts the target variable by looking at the 'k' closest neighbors in the dataset. It works by calculating the distance between the input data point and all the other data points(depends what the k value is, it's 4 in this project) using a distance metric(called Manhattan distance in this project), and then averaging their predictions. Because of that, it's very good for non-linear relationships, simple to implement, and very versatile and adaptable when given new and unseen data. However, it can overfit, especially when the k value is low, can be computationally expensive for large datasets, and can't handle high-dimensional data well.")

            st.markdown("### SVR (Support Vector Regression)")
            st.write("SVR is a type of machine learning algorithm that finds the best-fit function(hyperplane) within a certain margin of tolerance to predict the target variable while still trying to keep the model 'flat'. It works by mapping the input data into a higher-dimensional space using a kernel function and then finding the hyperplane that best fits the data while minimizing the error within the tolerance. This makes SVR very effective for high-dimensional data, can handle non-linear relationships well(if using the radial basis function kernel like in this project), and is robust to outliers. However, it's computationally expensive, sensitive to hyperparameters, and has limited interpretability and poor scalability.")

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
    crop_yield_testing_set = load_testing_data()
    crop_yield_training_set = load_training_data()
    crop_yield.rename(columns = {"Area":"Country", "Item":"Crop", "pesticides_tonnes":"pesticides_in_tons_used", "average_rain_fall_mm_per_year":"average_rainfall_in_mm_per_year"}, inplace=True)
    crop_yield = crop_yield.reindex(columns = ["Country", "Crop", "Year", "average_rainfall_in_mm_per_year", "pesticides_in_tons_used", "avg_temp", "hg/ha_yield"])

    st.markdown("<h1 style = 'text-align:center; font-size:50px; color:#FFFFFF;'><b>Datasets Used While Developing the Models🌱</b></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px; text-align:center; color:#FFFFFF; margin-top:-5px'>Here is the full dataset, training set, and testing set used during the development of the models. To ensure that the strongest models are performing well, you can test values from the testing set.</p>", unsafe_allow_html=True)
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Full Dataset", "Testing Set", "Training Set"])

    with tab1:
        st.write("This is the full dataset before stratification, consisting of 28,242 data points from Kaggle.")
        st.dataframe(crop_yield)

    with tab2:
        st.write("This is the testing set after stratification, consisting of 5,649 data points(20% of the full dataset) from the full dataset. This set is used to test the models and see how accurate they are when given unseen data.")
        st.dataframe(crop_yield_testing_set)

    with tab3:
        st.write("This is the training set after stratification, consisting of 22,593 data points(80% of the full dataset) from the full dataset. This set is used to train the models and make them as accurate as possible before being feeding them to the testing set.")
        st.dataframe(crop_yield_training_set)

def show_graphs_page():
    @st.cache_data
    def load_model_evaluation_data():
        return pd.read_csv("models_and_datasets/test_set_model_metrics.csv", index_col=0)

    @st.cache_data
    def get_all_predictions():
        testing_set = pd.read_csv("models_and_datasets/crop_yield_testing_set.csv", index_col=0)
        X_test = testing_set.drop(columns=["hg/ha_yield"])
        
        preds = {
            "Actual": testing_set["hg/ha_yield"],
            "K-Nearest Neighbors": loaded_final_k_nearest_neighbors_model.predict(X_test),
            "Random Forest Regression": loaded_final_random_forest_regression_model.predict(X_test),
            "Support Vector Regression": loaded_final_support_vector_regression_model.predict(X_test),
            "Gradient Boosting": loaded_final_gradient_boosting_regression_model.predict(X_test),
            "Polynomial Regression": loaded_final_polynomial_regression_model.predict(X_test),
            "Lasso Regression": loaded_final_lasso_regression_model.predict(X_test)
        }
        return pd.DataFrame(preds)

    st.markdown("<h1 style = 'text-align:center; font-size:50px; color:#FFFFFF;'><b>Graphs of Model Evaluation Metrics📈</b></h1>", unsafe_allow_html=True)
    st.markdown("<p style = 'font-size:18px; text-align:center; color:#FFFFFF; margin-top:-5px'>Here are the various types of graphs for all the models! This is to provide some visuals on how well the model performed through both the training and testing phases. </p>", unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Bar Graphs", "Scatter Plots", "Residual Plots"])
    model_evaluation_data = load_model_evaluation_data()

    all_preds = get_all_predictions()

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            rmse_mae_data = model_evaluation_data.loc[["RMSE", "MAE"]]

            rmse_mae_bar = px.bar(
                rmse_mae_data.T, 
                barmode="group",
                title="RMSE and MAE of the Models (Lower is Better)",
                color_discrete_sequence=["#2e7d32", "#81c784"],
                labels = {"variable": "Variables"}
            )
            rmse_mae_bar.update_layout(
                template="plotly_dark", 
                yaxis_title="Crop Yield in Hectograms/Hectare", 
                xaxis_title="Models",
                title = {"text": "RMSE and MAE of the Models (Lower is Better)", "x": 0.5, "xanchor": "center", "y": 0.85, "yanchor": "top"}
            )
            st.plotly_chart(rmse_mae_bar, use_container_width=True)

            mse_data = model_evaluation_data.loc[["MSE"]]
            mse_bar = px.bar(
                mse_data.T,
                title="MSE of the Models (Lower is Better)",
                color_discrete_sequence=["#2e7d32"],
                labels = {"variable":"Variable"}
            )
            mse_bar.update_layout(
                template="plotly_dark",
                yaxis_title="Crop Yield in Hectograms²/Hectare²",
                xaxis_title="Models",
                title = {"text": "MSE of the Models (Lower is Better)", "x": 0.5, "xanchor": "center", "y": 0.85, "yanchor": "top"}
            )
            st.plotly_chart(mse_bar, use_container_width=True)

        with col2:
            r2_data = model_evaluation_data.loc[["R^2"]]
            r2_bar = px.bar(
                r2_data.T, 
                title="R² of the Models (Higher is Better)",
                color_discrete_sequence=["#2e7d32"],
                labels = {"variable":"Variable"}
            )
            r2_bar.update_layout(
                template="plotly_dark", 
                yaxis_title="Score",
                xaxis_title="Models",
                title = {"text": "R² of the Models (Higher is Better)", "x": 0.5, "xanchor": "center", "y": 0.85, "yanchor": "top"}
            )
            st.plotly_chart(r2_bar, use_container_width=True)

    with tab2:
            model = st.selectbox("Select Model", models, key = "scatter_model_selectbox")
            fig = px.scatter(
                all_preds,
                x="Actual",
                y=model,
                template="plotly_dark",
                labels={"Actual": "Actual Yield", model: "Predicted Yield"},
                title = f"{model} Scatter Plot"
            )
            fig.update_layout(
                xaxis_title="Actual Yield (Hectograms/Hectare)",
                yaxis_title="Predicted Yield (Hectograms/Hectare)",
                title = {"text": f"{model} Scatter Plot", "x": 0.545, "xanchor": "center", "y": 0.85, "yanchor": "top"},
                autosize=True
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
            model = st.selectbox("Select Model", models, key = "residual_model_selectbox")
            residuals = all_preds["Actual"] - all_preds[model]
            fig = px.scatter(
                all_preds,
                x=model,
                y=residuals,
                template="plotly_dark",
                title = f"{model} Residual Plot",
                labels={model: "Predicted Yield", "y": "Residuals"}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title="Predicted Yield (Hectograms/Hectare)",
                yaxis_title="Residuals (Hectograms/Hectare)",
                title = {"text": f"{model} Residual Plot", "x": 0.545, "xanchor": "center", "y": 0.85, "yanchor": "top"},
                autosize=True
            )
            st.plotly_chart(fig, use_container_width=True)

pages = st.navigation([
    st.Page(show_predictor_page, title="Crop Yield Predictor", icon="🌽"),
    st.Page(show_statistics_page, title="Statistics", icon="📊"),
    st.Page(show_datasets_page, title="Datasets Used", icon="🌱"),
    st.Page(show_graphs_page, title="Graphs", icon="📈")
])
pages.run() 