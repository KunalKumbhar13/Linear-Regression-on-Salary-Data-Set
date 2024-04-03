#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Salary_Data.csv')


# In[3]:


print(df.sample(10))


# In[4]:


print(df.columns)


# In[5]:


print(df.shape)


# In[6]:


df.isnull().sum()


# In[16]:


X = df.iloc[:, :-1].values   #yearoexp
#Salary
y = df.iloc[:, 1].values


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


linear_regression = LinearRegression()


# In[21]:


linear_regression.fit(X_train, y_train)


# In[22]:


predictions = linear_regression.predict(X_test)


# In[23]:


print('Predicted             -    Original')
for pos in range(0, len(predictions)):
    print(f'{predictions[pos]:<{25}}  {y_test[pos]:<{15}}')


# In[27]:


plt.scatter(X_train, y_train, color='orange')
plt.plot(X_train, linear_regression.predict(X_train), color='silver')
plt.title('Years VS Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[28]:


plt.scatter(X_test, y_test, color='orange')
plt.plot(X_train, linear_regression.predict(X_train), color='silver')
plt.title('Years VS Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[29]:


#import library

from sklearn.metrics import mean_squared_error,r2_score

# model evaluation for training set

y_train_predict = linear_regression.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = linear_regression.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[30]:


# For finding slope for new regression line

Xm = np.mean(X_train)
Ym = np.mean(y_train)
sum1 = 0
sum2 = 0
print('Experience Salary     d=Xi-Xm               e=Yi-Ym             d*e                  e*e')
print('------------------------------------------------------------------------------------------------')
for pos in range(0, len(X_train)):
    d = (X_train[pos] - Xm)
    e = (y_train[pos] - Ym)
    sum1 = sum1 + d*e
    sum2 = sum2 = d*d
    print(f'{str(X_train[pos]):{10}} {str(y_train[pos]):{10}} {str(X_train[pos]-Xm):{20}} {str(y_train[pos]-Ym):20} {str(d*e):{20}} {str(d*d):{20}}')


# In[ ]:




