import pandas as pd
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, ExponentialFitter
import matplotlib
matplotlib.use('TkAgg')


data = pd.read_csv('telco.csv')
data['marital'] = data['marital'].map({'Married': 1, 'Unmarried': 0})
data['voice'] = data['voice'].map({'Yes': 1, 'No': 0})
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data['retire'] = data['retire'].map({'Yes': 1, 'No': 0})
data['internet'] = data['internet'].map({'Yes': 1, 'No': 0})
data['forward'] = data['forward'].map({'Yes': 1, 'No': 0})
data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})

# Keep all numeric data.
data = data.drop(columns=['custcat', 'region', 'ed', 'ID'])

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Explore data distributions
print(data.describe())

# Build AFT models with different distributions
weibull_model = WeibullAFTFitter()
weibull_model.fit(data, duration_col='tenure', event_col='churn')

loglogistic_model = LogLogisticAFTFitter()
loglogistic_model.fit(data, duration_col='tenure', event_col='churn')

exponential_model = ExponentialFitter()
exponential_model.fit(data['tenure'], event_observed=data['churn'])

# Model Comparison
print(weibull_model.summary)
print(loglogistic_model.summary)
print(exponential_model.summary)


plt.figure(figsize=(10, 6))
weibull_model.plot(label='Weibull')
loglogistic_model.plot(label='Log-Logistic')
exponential_model.plot(label='Exponential')
plt.legend()
plt.title('Survival Curves for AFT Models')
plt.xlabel('Tenure')
plt.ylabel('Cumulative Hazard')
plt.show()


final_model = loglogistic_model
# Calculate CLV
predicted_lifetime = final_model.predict_median(data)
print(predicted_lifetime)

# Calculate CLV for each customer
data['CLV'] = predicted_lifetime * data['tenure']

# Sum up CLV for all customers
total_clv = data['CLV'].sum()

print("Total CLV for all customers:", total_clv)




