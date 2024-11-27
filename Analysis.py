## Imports
# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

## Read in file
# %% read in data
filepath_combined = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/combined.csv'
wine_data_combined = pd.read_csv(filepath_combined)

filepath_red = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/winequality_red.csv'
wine_data_red = pd.read_csv(filepath_red, sep=';')

filepath_white = '/Users/cdcoonce/Documents/GitHub/Practice_Data_Sets/Wine Quality/Dataset/winequality_white.csv'
wine_data_white = pd.read_csv(filepath_white, sep=';')

## Exploration
# %% Descriptive Statistics
wine_data_combined.describe()
wine_data_red.describe()
wine_data_white.describe()

# %% Data Types
print(wine_data_combined.dtypes)
print(wine_data_red.dtypes)
print(wine_data_white.dtypes)

# %% Checking for Missing Values
print(wine_data_combined.isnull().sum())
print(wine_data_red.isnull().sum())
print(wine_data_white.isnull().sum())

# %% Distibution of Red wine data
wine_data_red.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distributions - Red Wine")
plt.show()

# %% Distibution of White wine data
wine_data_white.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distributions - White Wine")
plt.show()

# %% Pairplot features
key_features = ['fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'density', 'alcohol', 
                'quality']

sns.pairplot(wine_data_red[key_features], hue='quality', palette='coolwarm')
plt.suptitle("Pairplot of Red Wine Features")
plt.show()

sns.pairplot(wine_data_white[key_features], hue='quality', palette='coolwarm')
plt.suptitle("Pairplot of White Wine Features")
plt.show()

# %% Correlation Matrix
plt.figure(figsize=(12,10))

plt.subplot(2,1,1)
sns.heatmap(wine_data_red.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidth=0.5)
plt.title("Correlation Matrix - Red Wine", fontsize='16')

plt.subplot(2,1,2)
sns.heatmap(wine_data_white.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidth=0.5)
plt.title("Correlation Matrix - White Wine", fontsize='16')

plt.tight_layout()
plt.show()

# %% Removing highly correlated features
def remove_highly_correlated_features(df, target_column, threshold=0.65):
    correlation_matrix = df.corr()
    abs_correlation_matrix = correlation_matrix.abs()

    removed_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs_correlation_matrix.iloc[i,j] > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]

                if col1 in removed_features or col2 in removed_features:
                    continue

                if abs(correlation_matrix.loc[col1, target_column]) >= abs(correlation_matrix.loc[col2, target_column]):
                    removed_features.add(col2)
                else:
                    removed_features.add(col1)

    reduced_df = df.drop(columns=removed_features)

    return reduced_df, list(removed_features)

target_column = 'quality'
reduced_red_wine, removed_red_features = remove_highly_correlated_features(wine_data_red, target_column)
print("Removed Features for Red Wine:", removed_red_features)


reduced_white_wine, removed_white_features = remove_highly_correlated_features(wine_data_white, target_column)
print("Removed Features for White Wine:", removed_white_features)

# %% Train-Test Split
X_red = wine_data_red.drop('quality', axis=1)
y_red = wine_data_red['quality']

X_white = wine_data_white.drop('quality', axis=1)
y_white = wine_data_white['quality']

X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
    X_red, y_red, test_size=0.2, random_state=42
)

X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(
    X_white, y_white, test_size=0.2, random_state=42
)

print(f"Red Wine Train Shape: {X_red_train.shape}, Test Shape: {X_red_test.shape}")
print(f"White Wine Train Shape: {X_white_train.shape}, Test Shape: {X_white_test.shape}")

# %% Feature Importance with Random Forest

def plot_feature_importance(X_train, y_train, wine_type):
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    feature_importances = rf_model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {wine_type} Wine')
    plt.gca().invert_yaxis()
    plt.show()

    return importance_df

print("Feature Importance for Red Wine:")
red_importance = plot_feature_importance(X_red_train, y_red_train, "Red")

print("Feature Importance for White Wine:")
white_importance = plot_feature_importance(X_white_train, y_white_train, "White")

# %% Modeling
def train_and_evaluate_model(X_train, X_test, y_train, y_test, wine_type, n_estimators=100, random_state=42):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"{wine_type} Wine - Model Performance:")
    print(f"Training RMSE: {train_rmse:.2f}, Test_RMSE: {test_rmse:.2f}")
    print(f"Training R^2: {train_r2:.2f}, Test R^2: {test_r2:.2f}")

    return rf_model

# %% Full feature training
rf_red_full = train_and_evaluate_model(X_red_train, X_red_test, y_red_train, y_red_test, wine_type="Red")

rf_white_full = train_and_evaluate_model(X_white_train, X_white_test, y_white_train, y_white_test, wine_type="White")

# %% Train-Test Split
X_red_reduced = reduced_red_wine.drop('quality', axis=1)
y_red_reduced = reduced_red_wine['quality']

X_white_reduced = reduced_white_wine.drop('quality', axis=1)
y_white_reduced = reduced_white_wine['quality']

X_red_reduced_train, X_red_reduced_test, y_red_reduced_train, y_red_reduced_test = train_test_split(
    X_red_reduced, y_red_reduced, test_size=0.2, random_state=42
)

X_white_reduced_train, X_white_reduced_test, y_white_reduced_train, y_white_reduced_test = train_test_split(
    X_white_reduced, y_white_reduced, test_size=0.2, random_state=42
)

print(f"Red Wine Train Shape: {X_red_reduced_train.shape}, Test Shape: {X_red_reduced_test.shape}")
print(f"White Wine Train Shape: {X_white_reduced_train.shape}, Test Shape: {X_white_reduced_test.shape}")

# %% reduced feature training
rf_red_reduced = train_and_evaluate_model(X_red_reduced_train, X_red_reduced_test, y_red_reduced_train, y_red_reduced_test, wine_type="Red")

rf_white_reduced = train_and_evaluate_model(X_white_reduced_train, X_white_reduced_test, y_white_reduced_train, y_white_reduced_test, wine_type="White")

# %% Reduced Feature importance Plot
print("Feature Importance for Red Wine:")
red_reduced_importance = plot_feature_importance(X_red_reduced_train, y_red_reduced_train, "Red")

print("Feature Importance for White Wine:")
white_reduced_importance = plot_feature_importance(X_white_reduced_train, y_white_reduced_train, "White")

# %% Dropping Citric Acid
reduced_red_wine =  reduced_red_wine.drop('citric acid', axis=1)
reduced_white_wine =  reduced_white_wine.drop('citric acid', axis=1)

# %% Dropping more features
reduced_red_wine =  reduced_red_wine.drop(['density', 'residual sugar'], axis=1)
reduced_white_wine =  reduced_white_wine.drop(['sulphates', 'fixed acidity', 'chlorides'], axis=1)
# %% Dropping all but three features
reduced_red_wine =  reduced_red_wine.drop(['pH', 'chlorides', 'total sulfur dioxide'], axis=1)
reduced_white_wine =  reduced_white_wine.drop(['total sulfur dioxide', 'pH'], axis=1)

# %% Plotting residual - Red Wine
red_train_residuals = y_red_reduced_train - rf_red_reduced.predict(X_red_reduced_train)
red_test_residuals =  y_red_reduced_test - rf_red_reduced.predict(X_red_reduced_test)

# Plot residual distributions
plt.figure(figsize=(10, 6))
sns.histplot(red_train_residuals, kde=True, color="blue", label="Train Residuals")
sns.histplot(red_test_residuals, kde=True, color="red", label="Test Residuals")
plt.axvline(0, color="black", linestyle="--", label="Zero Residual")
plt.title("Residual Distribution - Red Wine")
plt.legend()
plt.show()

# Plot residuals vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(rf_red_reduced.predict(X_red_reduced_train), red_train_residuals, alpha=0.6, color="blue", label="Train")
plt.scatter(rf_red_reduced.predict(X_red_reduced_test), red_test_residuals, alpha=0.6, color="red", label="Test")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residuals vs. Predicted Values - Red Wine")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %% Plot Residuals - White Wine
white_train_residuals = y_white_reduced_train - rf_white_reduced.predict(X_white_reduced_train)
white_test_residuals =  y_white_reduced_test - rf_white_reduced.predict(X_white_reduced_test)

plt.figure(figsize=(10, 6))
sns.histplot(white_train_residuals, kde=True, color="blue", label="Train Residuals")
sns.histplot(white_test_residuals, kde=True, color="red", label="Test Residuals")
plt.axvline(0, color="black", linestyle="--", label="Zero Residual")
plt.title("Residual Distribution - White Wine")
plt.legend()
plt.show()

# Plot residuals vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(rf_white_reduced.predict(X_white_reduced_train), white_train_residuals, alpha=0.6, color="blue", label="Train")
plt.scatter(rf_white_reduced.predict(X_white_reduced_test), white_test_residuals, alpha=0.6, color="red", label="Test")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residuals vs. Predicted Values - White Wine")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %% Identifying Outliers
def identify_outliers(residuals, actual_values, threshold=2):
    std_residuals = np.std(residuals)
    outlier_mask = np.abs(residuals) > threshold * std_residuals
    outliers = pd.DataFrame({
        "Residuals": residuals[outlier_mask],
        "True_Quality": actual_values[outlier_mask]
    }).reset_index(drop=True)
    
    return outliers

# %% IDentifying Red Wine Outliers
red_train_outliers = identify_outliers(red_train_residuals, y_red_reduced_train)
red_test_outliers = identify_outliers(red_test_residuals, y_red_reduced_test)

white_train_outliers = identify_outliers(white_train_residuals, y_white_reduced_train)
white_test_outliers = identify_outliers(white_test_residuals, y_white_reduced_test)

print("Red Wine - Train Outliers:")
print(red_train_outliers)
print("\nRed Wine - Test Outliers:")
print(red_test_outliers)

print("\nWhite Wine - Train Outliers:")
print(white_train_outliers)
print("\nWhite Wine - Test Outliers:")
print(white_test_outliers)

# %% Outlier descriptive stats - Red Wine
red_data = reduced_red_wine.loc[red_train_outliers.index]
red_data.describe()
red_train_outliers.describe()

# %% Outlier Descriptive stats - White Wine
white_data = reduced_white_wine.loc[white_train_outliers.index]
white_data.describe()
white_train_outliers.describe()

# %% Alcohol outliers - Red
sns.scatterplot(data=reduced_red_wine, x="alcohol", y="quality")
plt.scatter(red_data["alcohol"], red_data["quality"], color='red')
plt.title("Red Wine")
# %% Alcohol outliers - White
sns.scatterplot(data=reduced_white_wine, x="alcohol", y="quality")
plt.scatter(white_data["alcohol"], white_data["quality"], color='red')
plt.title("White Wine")
# %% Volatile Acidity outliers - Red
sns.scatterplot(data=reduced_red_wine, x="volatile acidity", y="quality")
plt.scatter(red_data["volatile acidity"], red_data["quality"], color='red')
plt.title("Red Wine")
# %% Volatile Acidity outliers - White
sns.scatterplot(data=reduced_white_wine, x="volatile acidity", y="quality")
plt.scatter(white_data["volatile acidity"], white_data["quality"], color='red')
plt.title("White Wine")

# %% sulphates outliers - Red
sns.scatterplot(data=reduced_red_wine, x="sulphates", y="quality")
plt.scatter(red_data["sulphates"], red_data["quality"], color='red')
plt.title("Red Wine")
# %% sulphates outliers - White
sns.scatterplot(data=reduced_white_wine, x="sulphates", y="quality")
plt.scatter(white_data["sulphates"], white_data["quality"], color='red')
plt.title("White Wine")

# %%
sns.boxplot(data=reduced_red_wine[["volatile acidity", "chlorides", "alcohol", "sulphates"]])

# %%
sns.boxplot(data=reduced_white_wine[["volatile acidity", "chlorides", "alcohol", "sulphates"]])

# %%
sns.pairplot(red_train_outliers, diag_kind='kde')
# %%
sns.pairplot(white_train_outliers, diag_kind='kde')
# %%
