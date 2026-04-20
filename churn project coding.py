import pandas as pd
customer_data = pd.read_csv("C:\\Users\\MUTHARASI\\Downloads\\synthetic_customer_churn_dataset.csv")
customer_data
#%%
customer_data.shape
#%%
customer_data.isnull().sum()
#%%
customer_data.describe()
#%%
customer_data["PaymentMethod"].unique
#%%
customer_dataset = customer_data.sample(n = 200)
#%%
customer_dataset
#%%
customer_dataset['Contract'] = customer_dataset['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
#%%
from sklearn.model_selection import train_test_split
df = pd.DataFrame(customer_dataset)
X = df[["CustomerID","Tenure","Contract"]]
y = df[["Churn"]]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(X_test)
#%%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted:", y_pred)
#%%
from matplotlib import pyplot as plt
customer_dataset['CustomerID'].value_counts().head(10).plot(kind='bar')
plt.xlabel("Contract")
plt.ylabel("Churn")
plt.title("Contract VS Churn")
plt.show()
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

