Overview

This project aims to detect fraudulent credit card transactions using machine learning and deep learning techniques. The dataset used contains anonymized transaction data, with a binary classification label indicating whether a transaction is fraudulent or not.

Approach

1. Data Loading and Exploration

The dataset is loaded using Pandas and basic exploratory data analysis is performed.
The distribution of fraudulent vs. non-fraudulent transactions is visualized.

2. Data Preprocessing

Features are separated into X (input) and y (target).
Standardization is applied using StandardScaler to normalize features.
Class imbalance is addressed using SMOTE (Synthetic Minority Over-sampling Technique).
The dataset is split into training and testing sets.

3. Machine Learning Models

Logistic Regression:
A simple baseline model to compare performance.

Random Forest Classifier:
An ensemble learning method that improves predictions by combining multiple decision trees.

XGBoost Classifier:
A gradient boosting model optimized for high performance.

Anomaly Detection with Isolation Forest:
An unsupervised learning technique that detects anomalies based on data isolation.

4. Deep Learning Model:
LSTM (Long Short-Term Memory)
A deep learning model utilizing sequential data patterns.
The model consists of stacked LSTM layers with dropout to prevent overfitting.
The model is trained using the Adam optimizer and binary cross-entropy loss.

5. Evaluation Metrics:
Classification reports for all models.
Confusion matrix to analyze false positives and false negatives.
ROC-AUC Score to measure model performance.
Precision-Recall Curve to visualize trade-offs between precision and recall.

Results:
XGBoost achieved the highest performance with an improved ROC-AUC score.
The deep learning LSTM model further enhances fraud detection capabilities.
The implementation of SMOTE successfully balances the dataset and improves recall.

Future Improvements:
Further hyperparameter tuning for all models.
Implement additional deep learning models like CNNs or autoencoders.
Deploy the model into a real-time fraud detection system.

Dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost tensorflow

Usage:
python fraud_detection.py

Conclusion
This project demonstrates the effectiveness of machine learning and deep learning models in detecting fraudulent credit card transactions. Further optimization and deployment can enhance real-world applicability.

