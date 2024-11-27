#%% Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

#%% read in data
filepath_combined = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/combined.csv'
filepath_red= '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/combined.csv'
filepath_white = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/combined.csv'
wine_data_combined = pd.read_csv(filepath_combined)
wine_data_red = pd.read_csv(filepath_red)
wine_data_white = pd.read_csv(filepath_white)

# %% view structure
print(wine_data.head())
print(wine_data.info())

#  Check for missing data
print(wine_data.isnull().sum())

#  descriptive statistics
print(wine_data.describe())

#  Data Types
print(wine_data.dtypes)

# %% Visualizing Quality 
sns.countplot(x='quality', data=wine_data)
plt.title('Distibution of Wine Qaulity')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

#%% Separate Features(X) and targets(y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")


# %% Correlation Heatmap
correlation_matrix = wine_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# %% Filter features
treshhold = 0.2

correlation_with_target = wine_data.corr()['quality'].abs()
significant_features = correlation_with_target[correlation_with_target > treshhold].index
print(f"Significant features: {list(significant_features)}")

selected_features = list(significant_features.drop('quality'))
X_selected = X[selected_features]
print(f"Selected Features: {X_selected.columns}")

# %% Selected Features Correlation Heatmap
selected_features_correlation_matrix = X_selected.corr()
plt.figure(figsize=(8,6))
sns.heatmap(selected_features_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Between Selected Features')
plt.show()

# %% Feature Importance with RandomForest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_selected, y)

importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_selected.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# %% Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# %% Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %% Train the Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# %% Predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# %% Evaluate
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")

# %% Residual Analysis
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.scatter(y_train_pred, train_residuals, alpha=0.5, label='Train')
plt.scatter(y_test_pred, test_residuals, alpha=0.5, label='Test', color='red')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title("Residual Analysis")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %% Non-Linear Regression Model Train
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# %% Predict on the Train and Test sets
y_train_rf_pred = rf_model.predict(X_train)
y_test_rf_pred = rf_model.predict(X_test)

# %% Evaluate the model
train_rf_rmse = mean_squared_error(y_train, y_train_rf_pred, squared=False)
test_rf_rmse = mean_squared_error(y_test, y_test_rf_pred, squared=False)
train_rf_r2 = r2_score(y_train, y_train_rf_pred)
test_rf_r2 = r2_score(y_test, y_test_rf_pred)

print(f"Random Forest Training RMSE: {train_rf_rmse:.2f}, Test RMSE: {test_rf_rmse:.2f}")
print(f"Random Forest Training R²: {train_rf_r2:.2f}, Test R²: {test_rf_r2:.2f}")

# %% Residual Analysis
train_rf_residuals = y_train - y_train_rf_pred
test_rf_residuals = y_test - y_test_rf_pred

plt.scatter(y_train_rf_pred, train_rf_residuals, alpha=0.5, label='Train')
plt.scatter(y_test_rf_pred, test_rf_residuals, alpha=0.5, label='Test', color='red')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title("Residual Analysis for Random Forest")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %% Feature Importance
selected_features = ['volatile_acidity', 'chlorides', 'density', 'alcohol']
X_selected = X[selected_features]  # Subset the data to match selected features

# Generate the feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': selected_features,  # Use only the selected features
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # Flip to show the most important at the top
plt.show()

# %% Gradient Boosting Regressor
gbm_model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
gbm_model.fit(X_train, y_train)

# %% Predictions
y_train_gb_pred = gbm_model.predict(X_train)
y_test_gb_pred = gbm_model.predict(X_test)

# %% 
train_rmse = mean_squared_error(y_train, y_train_gb_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_gb_pred, squared=False)

train_r2 = r2_score(y_train, y_train_gb_pred)
test_r2 = r2_score(y_test, y_test_gb_pred)

print(f"Gradient Boosting Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Gradient Boosting Training R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")

# %% Residual Analysis
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred, train_residuals, color='blue', alpha=0.5, label='Train')
plt.scatter(y_test_pred, test_residuals, color='red', alpha=0.5, label='Test')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Analysis for Gradient Boosting')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# %% Feature Importance
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': gbm_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance from Gradient Boosting")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()

# %% Hyperparameter tuning for Random Forest Regression model
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=2,
    n_jobs=-1   # Use all available cores
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

# %% Best Model
best_rf = random_search.best_estimator_

# %% Evaluate on test data
y_test_pred = best_rf.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Test RMSE after tuning: {test_rmse}")

# %% Retraining the RF Model
best_rf_model = RandomForestRegressor(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=30,
    random_state=42
)

best_rf_model.fit(X_train, y_train)

# %% Predicting
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)


# %% Evaluate on the test set
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = best_rf_model.score(X_train, y_train)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = best_rf_model.score(X_test, y_test)

print(f"Training RMSE: {train_rmse:.2f}, Training R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.2f}")

# %% Residual Analysis
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=train_residuals, color="blue", label="Train")
sns.scatterplot(x=y_test_pred, y=test_residuals, color="red", label="Test")
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Residual Analysis for Best Random Forest Model")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %% Best Model Feature Importance
feature_importances = best_rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title("Feature Importance from Best Random Forest Model")
plt.show()

# %% Examine Residuals
X_train = pd.DataFrame(X_train, columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
X_test = pd.DataFrame(X_test, columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])

train_residuals = y_train - rf_model.predict(X_train)
test_residuals = y_test - rf_model.predict(X_test)

# Add residuals to the dataset for further analysis
train_data_with_residuals = X_train.copy()
train_data_with_residuals['Residuals'] = train_residuals
train_data_with_residuals['True_Quality'] = y_train
train_data_with_residuals['Predicted_Quality'] = rf_model.predict(X_train)

test_data_with_residuals = X_test.copy()
test_data_with_residuals['Residuals'] = test_residuals
test_data_with_residuals['True_Quality'] = y_test
test_data_with_residuals['Predicted_Quality'] = rf_model.predict(X_test)

# Visualize the residuals
plt.figure(figsize=(10, 6))
sns.histplot(test_residuals, kde=True, color='blue', alpha=0.6, bins=30)
plt.title("Distribution of Residuals (Test Data)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.axvline(x=0, color='black', linestyle='--', label='No Error')
plt.legend()
plt.show()

# %% Residuals by True Quality
# Residuals grouped by true wine quality
residual_by_quality = test_data_with_residuals.groupby('True_Quality')['Residuals'].agg(['mean', 'std']).reset_index()

# Plot residuals by quality
plt.figure(figsize=(10, 6))
sns.barplot(data=residual_by_quality, x='True_Quality', y='mean', palette='viridis')
plt.title('Average Residuals by True Wine Quality')
plt.xlabel('True Wine Quality')
plt.ylabel('Average Residual')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.show()

# %% Outlier Analysis
# Largest residuals in test data
outliers = test_data_with_residuals.loc[abs(test_data_with_residuals['Residuals']) > 1.5]
print("Outliers with Large Residuals:")
print(outliers[['Residuals', 'True_Quality', 'Predicted_Quality']].head(10))

# %% Correlation between residuals and features
# Correlation between residuals and features
correlation_with_residuals = test_data_with_residuals.corr()['Residuals'].sort_values()

# Plot correlation
plt.figure(figsize=(8, 6))
correlation_with_residuals[:-1].plot(kind='bar', color='coral')
plt.title('Correlation Between Features and Residuals')
plt.ylabel('Correlation')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.show()

# %%
