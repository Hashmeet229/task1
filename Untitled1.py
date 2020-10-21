#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:



url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[15]:



s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[16]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[17]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[18]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[19]:



line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[20]:


print(X_test) 
y_pred = regressor.predict(X_test) 


# In[21]:



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[27]:


hours = [9.25]
my_pred = regressor.predict([hours])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(my_pred[0]))


# In[ ]:





# In[ ]:




