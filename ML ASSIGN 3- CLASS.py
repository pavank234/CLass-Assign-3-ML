#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings # current version generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
df1=pd.read_csv("D:\PavanStudy\train.csv")
df1.head()


# In[2]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings # current version generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
df1=pd.read_csv("D:/PavanStudy/train.csv")
df1.head()


# In[3]:


le = preprocessing.LabelEncoder()
df1['Sex'] = le.fit_transform(df1.Sex.values)
df1['Survived'].corr(df1['Sex'])


# In[4]:


matrix = df1.corr()
print(matrix)


# In[5]:


df1.corr().style.background_gradient(cmap="Blues")


# In[6]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[8]:


#NAive bais

train_raw = pd.read_csv("D:/PavanStudy/train.csv")
test_raw = pd.read_csv("D:/PavanStudy/train.csv")

# Join data to analyse and process the set as one.
train_raw['train'] = 1
test_raw['train'] = 0
df1 = train_raw.append(test_raw, sort=False)




features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df1 = df1[features + [target] + ['train']]
# Categorical values need to be transformed into numeric.
df1['Sex'] = df1['Sex'].replace(["female", "male"], [0, 1])
df1['Embarked'] = df1['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df1.query('train == 1')
test = df1.query('train == 0')


# In[9]:


# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split, cross_validate
X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[10]:


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[11]:


#Question 2

glass=pd.read_csv("D:/PavanStudy/glass.csv")
glass.head()


# In[12]:


glass.corr().style.background_gradient(cmap="Oranges")


# In[13]:


sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[14]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


# In[15]:


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[16]:


from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[ ]:


#Which algorithm you got better accuracy? Can you justify why?

The Navie Bayes algorithm improves accuracy. Although Navie Bayes only needs a minimal quantity of training data, it frequently delivers good results for issues like spam detection and text classification.
SVM is more expensive than Navier-Bayes. For SVM, all data points in all dimensions must only have numerical values.

