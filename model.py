import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import warnings


# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
file_path = 'house.csv'
house_data = pd.read_csv(file_path)

# Perform exploratory data analysis (EDA)
eda_summary = {
    'head': house_data.head(),
    'info': house_data.info(),
    'describe': house_data.describe(),
    'null_values': house_data.isnull().sum()
}

# Displaying detailed information about the dataset
eda_summary['head']
eda_summary['info']
eda_summary['describe']
eda_summary['null_values']

# Check for missing values
missing_values = house_data.isnull().sum()

#Defining a function to hadle the outliers
def manage_outliers(df, column, method='IQR'):
    """
    Manage outliers in a dataframe column.

    Parameters:
    df (DataFrame): The Pandas DataFrame.
    column (str): The column name where to look for and manage outliers.
    method (str): Method to use for managing outliers. Default is 'IQR' (Interquartile Range).

    Returns:
    DataFrame: DataFrame with outliers managed.
    """
    df = df.copy()

    # Interquartile Range Method
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

# List of numerical columns to check for outliers
numerical_columns = ['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Price']

# Managing outliers for each numerical column
for col in numerical_columns:
    house_data = manage_outliers(house_data, col, method='IQR')

# Convert categorical features to one-hot encoded features
house_data_encoded = pd.get_dummies(house_data)

# Convert the data types of SqFt, Bedrooms, and Bathrooms to float so that MlFlow does not throw UserWarning
house_data_encoded = house_data_encoded.astype({'SqFt':'float64', 'Bedrooms':'float64', 'Bathrooms':'float64'})

# Visualizations
# Distribution of House Prices
plt.figure(figsize=(10, 6))
sns.histplot(house_data_encoded['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Recheck the box plots for numerical features after managing outliers
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
sns.boxplot(x=house_data_encoded['SqFt'], ax=axes[0, 0])
sns.boxplot(x=house_data_encoded['Bedrooms'], ax=axes[0, 1])
sns.boxplot(x=house_data_encoded['Bathrooms'], ax=axes[0, 2])
sns.boxplot(x=house_data_encoded['Offers'], ax=axes[1, 0])
sns.boxplot(x=house_data_encoded['Price'], ax=axes[1, 1])
axes[1, 2].set_visible(False) # Hide the last subplot as we have only 5 plots
plt.show()

# Compute the correlation matrix and plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(house_data_encoded.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(house_data)
plt.show()

# Count plots for categorical features
plt.figure(figsize=(12, 6))
sns.countplot(x='Brick', data=house_data)
plt.title('Count of Houses with and without Brick')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Neighborhood', data=house_data)
plt.title('Count of Houses in Different Neighborhoods')
plt.show()


# Initialize Linear Regression model
linear_reg = LinearRegression()

# Split the data into features and target variable
X = house_data_encoded.drop('Price', axis=1)
y = house_data_encoded['Price']

# EDA: Summary statistics and visualization
eda_summary = X.describe()

# Pairplot for visualizing relationships between features
sns.set_theme(style="ticks")
eda_pairplot = sns.pairplot(data=eda_summary, diag_kind="kde")

#Defining the test size and random state (hyperparameters)
test_size = 0.1
random_state = 40

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Set the experiment name
mlflow.set_experiment("House_Price_Prediction")

def train_model():
# Start an MLflow experiment
    mlflow.start_run()

    # Train the model
    linear_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = linear_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log EDA summary as a text artifact
    eda_summary_file = "eda_summary.txt"
    with open(eda_summary_file, "w") as eda_file:
      eda_file.write(eda_summary.to_string())
    mlflow.log_artifact(eda_summary_file)

    # Log the EDA pairplot as an image artifact
    pairplot_file = "eda_pairplot.png"
    eda_pairplot.savefig(pairplot_file)
    mlflow.log_artifact(pairplot_file)

    # Log the hyperparameters
    mlflow.log_param("Test size", test_size)
    mlflow.log_param("Random state", random_state)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic Linear Regression model for house.csv data")

    # Infer the model signature
    signature = infer_signature(X_train, linear_reg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=linear_reg,
        artifact_path="house_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-lr-model",
    )
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    column_names = list(house_data_encoded.columns)
    result = pd.DataFrame(X_test, columns=column_names)
    result["actual"] = y_test
    result["predicted"] = predictions
    result[:4]
    print(f"\n\nAccuracy for the model: {r2}.")
    # End the MLflow experiment
    mlflow.end_run()

# Train the model
if __name__ == "__main__":
    train_model()





