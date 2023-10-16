# %% [markdown]
# Import necssary libraries and modules

# %%
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score

# %% [markdown]
# Import training, validation and testing datasets

# %%
# Specify the file paths for the datasets
train_path = '/kaggle/input/audiomintest-layer12-dataset/train.csv'
valid_path = '/kaggle/input/audiomintest-layer12-dataset/valid.csv'
test_path = '/kaggle/input/audiomintest-layer12-dataset/test.csv'
# Load the training dataset
train_data = pd.read_csv(train_path)
# Load the validation dataset
valid_data = pd.read_csv(valid_path)
# Load the testing dataset
test_data = pd.read_csv(test_path)


# %% [markdown]
# Visualize original training data

# %%
train_data.head()

# %% [markdown]
# Prepare the data by eliminating any null values in the target labels and making educated guesses to fill in missing values in the features.
# 

# %% [markdown]
# Remove the columns in the training dataset where there are null values in the labels.

# %%
# Assess the presence of null values in the training dataset
train_null_counts = train_data.isnull().sum()
print("Null value counts in the training dataset: \n{}".format(train_null_counts))


# %% [markdown]
# Impute null values in the features of the train, valid, and test datasets by replacing them with their respective means.

# %%
# Replace null values with the mean in the training dataset
train_data = train_data.fillna(train_data.mean())

# Replace null values with the mean in the validation dataset
valid_data = valid_data.fillna(valid_data.mean())

# Replace null values with the mean in the test dataset
test_data = test_data.fillna(test_data.mean())


# %% [markdown]
# Visualize processed training data

# %%
train_data.head()

# %% [markdown]
# Separate features and labels in the train, valid and test datasets

# %%
# Split the train dataset into features and labels
train_features = train_data.iloc[:, :-4]
train_label1 = train_data.iloc[:, -4]

# Split the validation dataset into features and labels
valid_features = valid_data.iloc[:, :-4]
valid_label1 = valid_data.iloc[:, -4]

# # Split the test dataset into features and labels
# test_features = test_data.iloc[:, :-4]
# test_labels = test_data.iloc[:, -4:]
columns_to_remove = ['ID']
# Create a new DataFrame to hold the dropped column
saved_id_column = test_data[columns_to_remove].copy()

test_features = test_data.drop(columns=columns_to_remove)


# %% [markdown]
# Extract the first label in the train, valid and test datasets

# %%
# # Extract the first label from the training dataset
# train_label1 = train_labels.iloc[:, 0]

# # Extract the first label from the validation dataset
# valid_label1 = valid_labels.iloc[:, 0]

# # Extract the first label from the test dataset
# test_label1 = test_labels.iloc[:, 0]


# %% [markdown]
# # Making predictions for Label 1 performing feature engineering.

# %% [markdown]
# Predict label 1 with feature engineering steps and techniques

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# Visualize the distribution of the training label 1

# %%
# Visualize the distribution of the first target label in the training dataset
unique_labels, label_counts = np.unique(train_label1, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(unique_labels)
plt.bar(unique_labels, label_counts, color='lightcoral')
plt.xlabel('Label 1')
plt.ylabel('Count')
plt.title('Distribution of Label 1 in Training Data')
plt.show()


# %% [markdown]
# Calculate the correlation matrix of the training data features

# %%
# Compute the correlation matrix among the features
correlation_matrix = train_features.corr()

# Create a mask for the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix))

# Generate a heatmap to visualize the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='gray', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()


# %% [markdown]
# Identify the features that are highly correlated with each other using the traning dataset

# %%
# Define a correlation threshold
correlation_threshold = 0.9

# Initialize a set to store highly correlated feature pairs
highly_correlated_features = set()

# Identify highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            feature_name = correlation_matrix.columns[i]
            highly_correlated_features.add(feature_name)

print(highly_correlated_features)


# %% [markdown]
# Remove the previously identified highly correlated features from all the datasets

# %%
# # Eliminate features that are highly correlated
train_features = train_features.drop(columns=highly_correlated_features)
valid_features = valid_features.drop(columns=highly_correlated_features)
test_features = test_features.drop(columns=highly_correlated_features)


# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Show the number of features after filtering in the training dataset
print("Number of features after filtering in training data: {}".format(train_features.shape))

# Show the number of features after filtering in the validation dataset
print("Number of features after filtering in validation data: {}".format(valid_features.shape))

# Show the number of features after filtering in the test dataset
print("Number of features after filtering in test data: {}".format(test_features.shape))


# %% [markdown]
# Identify the features that are highly correlated with the label using the traning dataset

# %%
# Compute the correlation between features and the first target label
correlation_with_target = train_features.corrwith(train_label1)

# Define a correlation threshold
correlation_threshold = 0.05

# Identify features that have a correlation above the threshold with the target label
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)


# %% [markdown]
# Extract the features that are only highly correlated with the label from all datasets

# %%
# Remove features with low correlation from the training dataset
train_features = train_features[highly_correlated_features.index]

# Remove features with low correlation from the validation dataset
valid_features = valid_features[highly_correlated_features.index]

# Remove features with low correlation from the test dataset
test_features = test_features[highly_correlated_features.index]


# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Show the number of features after filtering in the training dataset
print("Number of features after filtering in training data: {}".format(train_features.shape))

# Show the number of features after filtering in the validation dataset
print("Number of features after filtering in validation data: {}".format(valid_features.shape))

# Show the number of features after filtering in the test dataset
print("Number of features after filtering in test data: {}".format(test_features.shape))


# %% [markdown]
# Standardize the features of all datasets

# %%
# Apply standardization to the features
feature_scaler = StandardScaler()
standardized_train_features = feature_scaler.fit_transform(train_features)
standardized_valid_features = feature_scaler.transform(valid_features)
standardized_test_features = feature_scaler.transform(test_features)


# %% [markdown]
# ### Feature Extraction

# %% [markdown]
# Identify and retain the most influential features for predicting the label through Principal Component Analysis (PCA).
# 
# Select features that collectively account for 99% of the variance in the label.

# %%
# Set the variance threshold to 99%
variance_threshold = 0.95

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

# Transform the standardized features using PCA
pca_train_result = pca.fit_transform(standardized_train_features)
pca_valid_result = pca.transform(standardized_valid_features)
pca_test_result = pca.transform(standardized_test_features)

# Get the explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_
print("Explained Variance Ratio after Dimensionality Reduction:", explained_variance_ratio_reduced)

# Plot the explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_result.shape[1] + 1), explained_variance_ratio_reduced, color='lightcoral')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component')
plt.show()

# Display the reduced feature matrix shapes
print("Shape of Reduced Train Feature Matrix: {}".format(pca_train_result.shape))
print("Shape of Reduced Validation Feature Matrix: {}".format(pca_valid_result.shape))
print("Shape of Reduced Test Feature Matrix: {}".format(pca_test_result.shape))


# %% [markdown]
# ## Model Selection

# %% [markdown]
# Select the model that best predicts the valid and test datasets based on accuracy, precision and recall

# %% [markdown]
# Hyper parameter tuning

# %%
Hyper Parameter Tuning For SVC

from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train_pca, y_train['label_1'])
grid.best_params_

# %% [markdown]
# Got the hyper parameters as C= 100,gamma=0.001

# %%
cross_val_score(SVC( C= 100,gamma=0.001), standardized_train_features, train_label1, cv=5, verbose=3).mean()

# %%
best_model_label_1 = SVC( C= 100,gamma=0.001)
pred_label1 = best_model_label_1.fit(standardized_train_features, train_label1).predict(standardized_test_features)
pred = best_model_label_1.predict(standardized_valid_features)
accuracy_score(valid_label1, pred )

# %% [markdown]
# # Generate Output CSV

# %% [markdown]
# Define method to create the csv file

# %%
def create_csv(pred):

  df = pd.DataFrame()

  df.insert(loc=0, column='ID', value=saved_id_column)
  df.insert(loc=1, column='Label 1', value=pred)
 
  df.to_csv('/kaggle/working/190676J_l12_label_1.csv', index=False)


# %% [markdown]
# Create CSV file

# %%
create_csv(pred_label1)


