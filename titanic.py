#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


gender_submission=pd.read_csv('gender_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:


gender_submission.head()


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


print(train.shape)
train.isnull().sum()


# In[8]:


print(test.shape)
test.isnull().sum()


# ## train dataset
train['Cabin'].unique()
# In[9]:


train.count()

survived_td = train[train.Survived==1]
survived_td.head()survived_td['Age'].unique()print('mean = ',survived_td['Age'].mean(),'\nmode = ',survived_td['Age'].mode(),'\nmedian = ',survived_td['Age'].median(),'\nstd = ',survived_td['Age'].std())survived_td['Age'].isnull().sum()
# In[10]:


train.columns


# In[11]:


train.head(1)


# In[12]:


x=train.drop('Survived',axis = 'columns')


# In[13]:


x.head()


# # .

# In[14]:


x.columns


# In[15]:


#plt.scatter(x=train['Survived'],y=x['Pclass'])
#plt.scatter(x=train['Survived'],y=x['Age'])
#plt.scatter(x=train['Survived'],y=x['SibSp'])
#plt.scatter(x=train['Survived'],y=x['Parch'])
#plt.scatter(x=train['Survived'],y=x['Fare'])
#plt.scatter(x=train['Survived'],y=x['Embarked'])
sns.pairplot(train,dropna=True,kind='reg')


# In[16]:


X=x.drop(['PassengerId','Name','Ticket','Cabin'],axis='columns')
X.head()


# In[17]:


## checking the values less than 1 so that we can convert back to the normal age
### assusme age in 100 ---> we will multiply age age less than one by  100 for getting age under 100

X[X['Age']<1.0]


# In[18]:


age = X.Age
age1=[]
for i in age:
    if i<1:
        i=i*100
        age1.append(i)
    else:
        age1.append(i)
age1
age2 = pd.DataFrame(age1)
age2.columns = ['age']
age2.shape


# In[19]:


x.shape


# In[21]:


X1 = X.assign(Age = age2)
X1.head()


# #### .
s = train[(train['Survived']==1) & (train['Age'].isnull())].shape
sns = train[(train['Survived']==0) & (train['Age'].isnull())].shape
nstrain.shape[0] - (train[(train['Survived']==1) & (train['Age'].isnull())].shape[0] + train[(train['Survived']==0) & (train['Age'].isnull())].shape[0])

# #### .
X1.Age.std()*2+X1.Age.mean()
# In[43]:


from sklearn.metrics import confusion_matrix


# In[ ]:





# In[44]:


X2 = X1.dropna()
X2.head()


# ###### convert catagories into the num

# In[45]:


from sklearn.preprocessing import LabelEncoder


# In[46]:


le=LabelEncoder()


# In[47]:


sex=le.fit_transform(X2.Sex)
sex.size


# In[48]:


embarked = le.fit_transform(X2.Embarked)
embarked.size


# In[49]:


X3 = X2.assign(Sex=sex,Embarked=embarked)
X3.head()

X3['Fare'] = X3['Fare'].round(2)
X3.sort_values(by=['Fare','age']).head()X3[X3.Fare==0]train[(train.Survived) & (train.Fare==0)]train[(train.Fare==0) & (~train.Age.isnull())]train.Embarked.unique()X4=X3.drop('Age',axis='columns')
X4.head()
# ### traing data cleaned properlly (X4 is final dataset)
# 
# ### Test dataset

# In[82]:


test.isnull().sum()


# In[83]:


test.Age.unique()

[test.Age[test.Age<1]*100]
# In[84]:


x_test=test.drop(['PassengerId','Name','Ticket','Cabin'],axis='columns')
x_test.shape


# In[85]:


x_test.head()


# In[86]:


age_test = x_test.Age
age_test1=[]
for i in age_test:
    if i<1:
        i=i*100
        age_test1.append(i)
    else:
        age_test1.append(i)    
age_test1
age_test2 = pd.DataFrame(age_test1)
age_test2.columns = ['age']
age_test2.shape

age_test2=age_test2[~(age_test2.age<1)]
age_test2
# In[87]:


X_test=x_test.assign(Age=age_test2)
X_test.head()


# In[88]:


X_test.isnull().sum()


# In[101]:


X_test.Age.unique()


# In[103]:


X_test.Age.size


# In[107]:


mode = X_test.Age.mode().to_list()
mean = X_test.Age.mean()
std = X_test.Age.std()
rand = std*2+mean


# In[173]:


X_test.Fare.mode()


# In[175]:


X_test.Fare.value_counts()


# In[ ]:





# In[158]:


X_test.Age.value_counts().head(11)


# In[164]:


X_test.Age.fillna(value=np.random.randint(24,30),inplace=True)


# In[176]:


X_test.Fare.fillna(value = 7.7500,inplace=True)


# In[178]:


X_test.isnull().sum()


# In[179]:


sex_test = le.fit_transform(X_test.Sex)
sex_test.shape


# In[180]:


embarked_test = le.fit_transform(X_test.Embarked)
embarked_test.shape


# In[181]:


X_test1 = X_test.assign(Sex = sex_test,Embarked = embarked_test)
X_test1.head()


# ### test dataset is cleaned properlly  and ( X_test1 ) final result
# 
# #### Ready for data cleaning 

# In[184]:


print('shape = ',X_test1.shape)
X_test1.head()


# In[186]:


print('shape = ',X3.shape)
X3.head()


# ### for test dataset for dv variables

# In[187]:


y_test=gender_submission
y_test.head()


# In[188]:


y_test0=y_test.drop('PassengerId',axis='columns')

y_test0.head()


# ### train dataset for idv

# In[189]:


y_train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis= 'columns')
y_train.isnull().sum()


# In[190]:


dv = y_train.dropna()
dv = pd.DataFrame(dv.Survived)


# In[191]:


print(X3.shape)
dv.shape


# In[192]:


dv.head()


# In[193]:


dv_train = dv
idv_train = X3
dv_test = y_test0
idv_test = X_test1


# In[194]:


print('dv_train = ',dv_train.shape,', idv_train = ',idv_train.shape,'\ndv_test = ',dv_test.shape,', idv_test = ',idv_test.shape)


# In[195]:


idv_test.tail()

train(11) --> { except survived }  ==>
test(1) ---> { survived }
# ### training the data using the machine learning 
# 
# #### DecisiontreeClassifier, RandomForestreClassifier, SVM, LogisticRegression 

# In[196]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf_model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)
features=idv_train.columns
rf_model.fit(idv_train,dv_train)
print('oob accuracy = ',rf_model.oob_score_)for features,imp in zip(features,rf_model.feature_importances_):
    print(features,imp)
# In[247]:


lr = LogisticRegression(solver='newton-cg')
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=1000,max_features=5,oob_score=True)
svm = SVC(kernel='linear',gamma='auto')


# In[248]:


models = [lr,dt,rf,svm]
model_name = ['logistic','decision','random forest','svm']
for i in range(len(models)):
    
    #models[i].fit(idv_train,dv_train)
    print(model_name[i],'score =',models[i].fit(idv_train,dv_train).score(idv_train,dv_train))


# In[249]:


lr_pred = lr.predict(idv_test)
lr_pred.size


# In[250]:


pred_model = []
for i in range(len(models)):
    
    print(model_name[i],'score =',models[i].fit(idv_train,dv_train).score(idv_train,dv_train))
    pred_model.append(pd.DataFrame([models[i].fit(idv_train,dv_train).predict(idv_test)]))


# In[255]:


pred_modl = pd.concat([pred_model[0],pred_model[1],pred_model[2],pred_model[3]])


# In[256]:


pred_modl = pred_modl.T


# In[257]:


pred_modl.head(3)


# In[258]:


pred_modl.columns = model_name
print(pred_modl.shape)
pred_modl.head()

pred_modl.reset_index(drop=True,inplace=True)
gender_submission.reset_index(drop=True,inplace=True)
# In[259]:


check = pd.concat([gender_submission,pred_modl],axis='columns')
check.head(10)


# ### checking the how much it predicted properly using confusion matrix

# In[260]:


from sklearn.metrics import confusion_matrix


# In[270]:


lr_c_matrix = confusion_matrix(check.Survived,check.logistic)
print(lr_c_matrix)
sns.heatmap(lr_c_matrix, annot=True)


# In[271]:


dt_c_matrix = confusion_matrix(check.Survived,check.decision)
print(dt_c_matrix)
sns.heatmap(dt_c_matrix, annot=True)


# In[272]:


rf_c_matrix = confusion_matrix(check.Survived,check['random forest'])
print(rf_c_matrix)
sns.heatmap(rf_c_matrix, annot=True)


# In[273]:


svm_c_matrix = confusion_matrix(check.Survived,check.svm)
print(svm_c_matrix)
sns.heatmap(svm_c_matrix, annot=True)


# ## .

# # SVM worked properly and predicted accurately
# ### parameter used in svm is { kernal = 'linear', gamma = 'auto'}
# 

# In[ ]:




