

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC

import xgboost as xgb

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report, confusion_matrix

import warnings

import pickle

import pandas as pd

df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
df.head()

df.columns

df.drop(['isFlaggedFraud'],axis =1,inplace = True)

df

df.head()

df.tail()

import matplotlib.pyplot as plt
import warnings
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

df_numeric = df.select_dtypes(include=['number'])
df_numeric.corr()

import seaborn as sns
sns.heatmap(df_numeric.corr(), annot=True)

sns.histplot(data=df,x='step')

sns.boxplot(data=df,x='step')

sns.countplot(data=df,x='type')

sns.histplot(data=df,x='amount')

sns.boxplot(data=df,x='amount')

sns.histplot(data=df,x='oldbalanceOrg')

df['nameDest'].value_counts()

sns.boxplot(data=df,x='oldbalanceDest')

sns.boxplot(data=df,x='newbalanceDest')

sns.countplot(data=df,x='isFraud')

df['isFraud'].value_counts()

df.loc[df['isFraud']==0,'isFraud']='is not Fraud'
df.loc[df['isFraud']==1,'isFraud']='is Fraud'

df

sns.jointplot(data=df,x='newbalanceDest',y='isFraud')

sns.countplot(data=df,x='type',hue='isFraud')

sns.boxplot(data=df,x='isFraud',y='step')

sns.boxplot(data=df,x='isFraud',y='amount')

sns.boxplot(data=df,x='isFraud',y='oldbalanceOrg')

sns.boxplot(data=df,x='isFraud',y='newbalanceOrig')

sns.violinplot(data=df,x='isFraud',y='oldbalanceDest')

sns.violinplot(data=df,x='isFraud',y='newbalanceDest')

df.describe(include='all')

df.info()

sns.boxplot(df['amount'])

from scipy import stats

print(stats.mode(df['amount']))

print(np.mean(df['amount']))

q1 = np.quantile(df['amount'],0.25)

q3 = np.quantile(df['amount'],0.75)

IQR = q3-q1

upper_bound = q3+(1.5 * IQR)

lower_bound = q1-(1.5 * IQR)

print('q1:',q1)

print('q3:',q3)

print('IQR:', IQR)

print('Upper Bound: ', upper_bound)

print('Lower Bound: ', lower_bound)

print('Skewed data : ', len(df [df['amount']>upper_bound]))

print('Skewed data : ',len(df [df['amount']<lower_bound]))

def transformationPlot(feature):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(feature, kde=True)
    plt.subplot(1,2,2)
    stats.probplot(feature, plot=plt)

transformationPlot(np.log(df['amount']))

df['amount']=np.log(df['amount'])

from sklearn.preprocessing import LabelEncoder

la= LabelEncoder()

df['type']=la.fit_transform(df['type'])

df['type'].value_counts()

df_cleaned = df.dropna()
x = df_cleaned.drop('isFraud', axis=1)
y=df_cleaned['isFraud']

x

y

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=0,test_size=0.2)

print(x_train.shape)

print(x_test.shape)

print(y_test.shape)

print(y_train.shape)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier()

# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed = x_test.drop(['nameOrig', 'nameDest'], axis=1)

rfc.fit(x_train_processed,y_train)

y_test_predict1=rfc.predict(x_test_processed)

test_accuracy=accuracy_score(y_test,y_test_predict1)

test_accuracy

y_train_predict1=rfc.predict(x_train_processed)

train_accuracy=accuracy_score(y_train,y_train_predict1)

train_accuracy

pd.crosstab(y_test,y_test_predict1)

print(classification_report(y_test,y_test_predict1))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed_dtc = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed_dtc = x_test.drop(['nameOrig', 'nameDest'], axis=1)

dtc.fit(x_train_processed_dtc, y_train)

y_test_predict2=dtc.predict(x_test_processed_dtc)

test_accuracy=accuracy_score(y_test,y_test_predict2)

test_accuracy

y_train_predict2=dtc.predict(x_train_processed_dtc)

train_accuracy=accuracy_score(y_train,y_train_predict2)

train_accuracy

pd.crosstab(y_test,y_test_predict2)

print(classification_report(y_test,y_test_predict2))

from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier()

# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed_etc = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed_etc = x_test.drop(['nameOrig', 'nameDest'], axis=1)

etc.fit(x_train_processed_etc,y_train)

y_test_predict3=etc.predict(x_test_processed_etc)

test_accuracy = accuracy_score(y_test,y_test_predict3)

test_accuracy

y_train_predict3=etc.predict(x_train_processed_etc)

train_accuracy= accuracy_score(y_train,y_train_predict3)

train_accuracy= train_accuracy

pd.crosstab(y_test,y_test_predict3)

print(classification_report(y_test,y_test_predict3))

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
svc=SVC()
# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed_svc = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed_svc = x_test.drop(['nameOrig', 'nameDest'], axis=1)

svc.fit(x_train_processed_svc,y_train)
y_test_predict4=svc.predict(x_test_processed_svc)
test_accuracy=accuracy_score(y_test,y_test_predict4)

test_accuracy

y_train_predict4=svc.predict(x_train_processed_svc)

train_accuracy=accuracy_score(y_train,y_train_predict4)

train_accuracy

pd.crosstab(y_test,y_test_predict4)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_test_predict4))

df.columns

from sklearn.preprocessing import LabelEncoder

la=LabelEncoder()

y_train1=la.fit_transform(y_train)

y_test1=la.transform(y_test)

y_test1-la.transform(y_test)

y_test1

y_train1

import xgboost as xgb

xgb1=xgb.XGBClassifier()

# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed_xgb = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed_xgb = x_test.drop(['nameOrig', 'nameDest'], axis=1)

xgb1.fit(x_train_processed_xgb, y_train1)

y_test_predict5 = xgb1.predict(x_test_processed_xgb)

test_accuracy=accuracy_score(y_test1,y_test_predict5)

test_accuracy

y_train_predict5=xgb1.predict(x_train_processed_xgb)

train_accuracy=accuracy_score(y_train1,y_train_predict5)

train_accuracy

pd.crosstab(y_test1,y_test_predict5)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test1,y_test_predict5))

def compareModel():

  print("train accuracy for rfc", accuracy_score(y_train_predict1,y_train))
  print("test accuracy for rfc", accuracy_score(y_test_predict1,y_test))

  print("train accuracy for dtc", accuracy_score(y_train_predict2,y_train))

  print("test accuracy for dtc", accuracy_score(y_test_predict2,y_test))
  print("train accuracy for etc", accuracy_score(y_train_predict3,y_train))

  print("test accuracy for etc", accuracy_score(y_test_predict3,y_test))

  print("tran accuracy for svc", accuracy_score(y_train_predict4,y_train))

  print("test accuracy for svcc", accuracy_score(y_test_predict4,y_test))
  print("train accuracy for xgb1", accuracy_score(y_train_predict5,y_train1))

  print("test accuracy for xgb1", accuracy_score(y_test_predict5,y_test1))

compareModel()

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

svc = SVC()

# Drop the 'nameOrig' and 'nameDest' columns from x as they are non-numeric identifiers
x_train_processed_svc = x_train.drop(['nameOrig', 'nameDest'], axis=1)
x_test_processed_svc = x_test.drop(['nameOrig', 'nameDest'], axis=1)

svc.fit(x_train_processed_svc,y_train)

y_test_predict4=svc.predict(x_test_processed_svc)

test_accuracy= accuracy_score(y_test,y_test_predict4)

test_accuracy

y_train_predict4=svc.predict(x_train_processed_svc)

train_accuracy=accuracy_score(y_train,y_train_predict4)

train_accuracy

import pickle

pickle.dump(svc,open('payments.pkl', 'wb'))