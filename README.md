# Credit Card Fraud Detector

## Overview
This post provides a detailed description of the Credit Card Fraud Detector project, including its purpose, the machine learning models used, and a comprehensive overview of the solution. The project aims to identify fraudulent credit card transactions using advanced machine learning techniques.

## Project Definition and Scope
In this solution, we build the core of a credit card fraud detection system using SageMaker. We start by training an unsupervised anomaly detection algorithm, Random Cut Forest (RCF), and then proceed to train two XGBoost models for supervised training. To deal with the highly imbalanced data common in fraud detection, our first model uses XGBoost's weighting schema, and the second uses a re-sampling technique, SMOTE, for oversampling the rare fraudulent examples. Lastly, we train an optimal XGBoost model with Hyper-parameter Optimization (HPO) to further improve model performance.

![image](https://github.com/user-attachments/assets/5a35ec8f-d82b-43a3-b93e-a9ff1e84be11)

## What is the Credit Card Fraud Detector?
The Credit Card Fraud Detector is a machine learning-based solution designed to identify fraudulent credit card transactions in real-time. By leveraging historical transaction data, the system can recognize patterns and anomalies that indicate potential fraud. The solution aims to reduce financial losses and enhance the security of credit card transactions.

The credit card fraud detector solution contains five stages:

### Stage I: Investigate and Process the Data
Set up the environment and process the dataset, which contains only numerical features transformed using PCA to protect user privacy. The dataset contains 28 PCA components (V1-V28), and two features that haven't been transformed: Amount and Time. The class column corresponds to whether a transaction is fraudulent. Given the class imbalance, with only a small fraction of data corresponding to fraudulent examples, we split the dataset into training and testing sets before applying techniques to alleviate class imbalance.

### Stage II: Train an Unsupervised Random Cut Forest Model
In a fraud detection scenario, labeling fraudulent examples takes time. Anomaly detection helps identify anomalous examples based solely on their feature characteristics. We fit the RCF model, deploy it, and evaluate its performance in separating fraudulent from legitimate transactions based on anomaly scores. High anomaly scores typically indicate fraudulent transactions.

### Stage III: Train a XGBoost Model with the Built-in Weighting Schema
We use a supervised learning algorithm, XGBoost, which discovers relationships between features and the dependent class. By specifying the XGBoost algorithm and using SageMaker's built-in XGBoost containers, we train the model with a focus on scaling the weights of the positive vs. negative class examples. This is crucial in imbalanced datasets to prevent the majority class from dominating the learning.

### Stage IV: Train a XGBoost Model with the Over-sampling Technique SMOTE
To address imbalanced problems, we use the Synthetic Minority Over-sampling Technique (SMOTE), which oversamples the minority class by interpolating new data points. We train the XGBoost model using the SMOTE dataset and evaluate its performance. Adjusting the classification threshold can balance between minimizing false positives and not missing any fraudulent cases.

### Stage V: Train an Optimal XGBoost Model with Hyper-parameter Optimization (HPO)
We further improve model performance using Hyper-parameter Optimization (HPO). The HPO process selects an optimal model based on performance on validation data. We prepare the input data in CSV format, combine the target variable with feature variables, and upload them to S3 buckets for training.

### Stage VI: Evaluate and Compare All Model Performances
We evaluate and compare the performances of all models. This step is critical to ensure the effectiveness of the fraud detection system, as it directly impacts both the security and customer experience.

#### False Negatives
- **Definition**: Transactions that are fraudulent but are incorrectly classified as legitimate.
- **Impact**: The primary concern here is the system's inability to spot new fraud patterns soon enough. This can lead to financial losses and undermine the trust in the detection system.
- **Considerations**: Minimizing false negatives is crucial to ensure that fraudulent activities are promptly identified and mitigated.

#### False Positives
- **Definition**: Transactions that are legitimate but are incorrectly classified as fraudulent.
- **Impact**: Blocking legitimate customers can lead to dissatisfaction, loss of customer trust, and potential financial loss due to interrupted transactions.
- **Considerations**: Reducing false positives is essential to maintain a smooth customer experience and prevent legitimate users from being mistakenly flagged.

#### Trade-off Management

Evaluating models for fraud detection involves managing the trade-off between false negatives and false positives. The key metrics used in this stage, such as the balanced accuracy, Cohen's Kappa score, F1 score, and ROC_AUC, help in assessing this trade-off:

- **Balanced Accuracy**: Accounts for both types of errors, providing a more comprehensive view of model performance.
- **Cohen's Kappa Score**: Measures the agreement between predicted and actual classifications, considering the possibility of chance agreements.
- **F1 Score**: Balances precision (minimizing false positives) and recall (minimizing false negatives), offering a single metric to evaluate overall performance.
- **ROC_AUC**: Provides insight into the trade-off between the true positive rate and the false positive rate across different threshold settings.

![WhatsApp Image 2024-07-12 at 09 32 55_49611bb7](https://github.com/user-attachments/assets/7114465f-af93-4add-ab7f-03e333bae721)

## Conclusion
In this solution, we cover data investigation, unsupervised anomaly detection, and multiple supervised learning techniques to build a robust credit card fraud detection system. By leveraging SageMaker's capabilities, we streamline the training, deployment, and evaluation processes to achieve optimal model performance.
