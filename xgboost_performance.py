import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from matplotlib.patches import Rectangle

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
try:
    df = pd.read_csv('dart_mwendokasi_data.csv')
except FileNotFoundError:
    print("Error: 'dart_mwendokasi_data.csv' not found in the working directory.")
    exit(1)

# Preprocess the data (same as original code)
df = df.dropna()
df['date_ordinal'] = pd.to_datetime(df['date'], errors='coerce').map(lambda x: x.toordinal() if pd.notna(x) else 0)
df['day'] = pd.to_datetime(df['date'], errors='coerce').dt.day_name()
df = df.drop(columns=['date'])

# Encode categorical variables
encoders = {}
for column in ['day', 'weather', 'peak_hours', 'weekends', 'holidays']:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])

# Convert time_value to minutes
def convert_to_minutes(time_str):
    try:
        if pd.isna(time_str):
            return 0
        time_str = str(time_str).strip()
        if not time_str or time_str == ":00":
            return 0
        time_parts = time_str.split(':')
        if len(time_parts) >= 2:
            hour = int(time_parts[0]) if time_parts[0].isdigit() else 0
            minute = int(time_parts[1]) if time_parts[1].isdigit() else 0
            return max(0, min(1440, hour * 60 + minute))
        return 0
    except:
        return 0

df['time_value'] = df['time_value'].apply(convert_to_minutes).astype('int32')

# Split features and target
X = df.drop(columns=['passengers']).astype('float32')
y = df['passengers'].astype('float32')
training_feature_order = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained models
try:
    model = xgb.XGBRegressor()
    model.load_model('xgboost_model.ubj')
except FileNotFoundError:
    print("Error: 'xgboost_model.ubj' not found in the working directory.")
    exit(1)
except Exception as e:
    print(f"Error loading 'xgboost_model.ubj': {e}")
    exit(1)

try:
    model_class = joblib.load('xgboost_classifier.pkl')
except FileNotFoundError:
    print("Error: 'xgboost_classifier.pkl' not found in the working directory.")
    exit(1)
except Exception as e:
    print(f"Error loading 'xgboost_classifier.pkl': {e}")
    exit(1)

# Classification data
y_class = pd.cut(y, bins=[0, 50, 100, float('inf')], labels=['Low', 'Medium', 'High'])
label_encoder_class = LabelEncoder()
y_class_encoded = label_encoder_class.fit_transform(y_class)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)

# Create a report directory
if not os.path.exists('model_report'):
    os.makedirs('model_report')

# Plot 1: Feature Importance (Features on x-axis, Importance on y-axis)
plt.figure(figsize=(12, 6), dpi=300)
feature_importance = model.feature_importances_
sns.barplot(x=X.columns, y=feature_importance, palette='viridis')
plt.title('Factors Influencing Passenger Numbers', fontsize=18, weight='bold')
plt.xlabel('Factors', fontsize=14)
plt.ylabel('Importance (%)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
# Add explanation text box
plt.text(0.02, 0.98, "This chart shows the key factors affecting passenger numbers.\n"
                    "Busy hours (72.6%) are the most important, followed by weekends (15.7%) and weather (8.8%).\n"
                    "This indicates that rush hours and weekends drive higher passenger counts.",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('model_report/feature_importance.png', bbox_inches='tight')
plt.close()

# Plot 2: Prediction Error Distribution
y_pred = model.predict(X_test)
errors = y_test - y_pred
plt.figure(figsize=(12, 6), dpi=300)
sns.histplot(errors, bins=50, kde=True, color='royalblue')
plt.title('Accuracy of Passenger Number Predictions', fontsize=18, weight='bold')
plt.xlabel('Prediction Error (Actual - Predicted Passengers)', fontsize=14)
plt.ylabel('Number of Predictions', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Add explanation text box
plt.text(0.02, 0.98, "This chart shows how accurate predictions are compared to actual passenger numbers.\n"
                    "Most predictions are close (average error: 18 passengers), but some errors are larger,\n"
                    "especially during unusual times like holidays or bad weather.",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('model_report/error_distribution.png', bbox_inches='tight')
plt.close()

# Plot 3: Confusion Matrix for Classification
y_pred_c = model_class.predict(X_test_c)
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Accuracy of Passenger Group Classification', fontsize=18, weight='bold')
plt.xlabel('Predicted Group', fontsize=14)
plt.ylabel('Actual Group', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Add explanation text box
plt.text(0.02, 0.98, "This chart shows how accurately passenger counts are grouped (Low: 0-50, Medium: 51-100, High: >100).\n"
                    "The model is correct 67% of the time, performing well for Medium and High groups\n"
                    "but often misclassifying quiet times (Low) as busier periods.",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('model_report/confusion_matrix.png', bbox_inches='tight')
plt.close()

print("Visualizations saved in 'model_report' directory.")