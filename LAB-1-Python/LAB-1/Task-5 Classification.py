# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] =7,5
rcParams['font.size'] = 11.0

from matplotlib import font_manager as fm
import sklearn.feature_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Loading the dataset
df = pd.read_csv('./datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Replacing empty values with NAN
df['TotalCharges'].replace(" ",np.nan,inplace=True)
# Dropping NaN values
df.dropna(inplace=True)

# Changing the datatype to float values
df['TotalCharges']=df['TotalCharges'].astype('float')

# Dropping the customerID column
df.drop(columns={'customerID'},inplace=True)

# Finding the correlation and displaying the heatmap plot
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':14})
heat_map=plt.gcf()
heat_map.set_size_inches(50,25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
plt.show()

# Converting the categorical data to the numerical data
telcom_df =  df.iloc[:,:19]
telcom_df['gender']=telcom_df.gender.apply(lambda x: 1 if x=='Male' else 0)
telcom_df['Partner']=telcom_df.Partner.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['Dependents']=telcom_df.Dependents.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['PhoneService']=telcom_df.PhoneService.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['PaperlessBilling']=telcom_df.PaperlessBilling.apply(lambda x: 1 if x=='Yes' else 0)

telcom_df =pd.get_dummies(data = telcom_df,columns = [
'MultipleLines',
'InternetService',
'OnlineSecurity',
'OnlineBackup',
'DeviceProtection',
'TechSupport',
'StreamingTV',
'StreamingMovies',
'Contract',
'PaymentMethod'],dtype=np.int32 )

# Dropping the columns which have very less correlation Observed from the heat map which is displayed from the abaove graph
telcom_df.drop(columns={'MultipleLines_No phone service','OnlineSecurity_No internet service'\
                   ,'OnlineBackup_No internet service','DeviceProtection_No internet service'\
                   ,'TechSupport_No internet service','StreamingTV_No internet service'\
                  ,'StreamingMovies_No internet service'},inplace=True)


for x in telcom_df.columns:
    if(x!='customerID' and x!='MonthlyCharges' and x!='tenure'):
        print(x,sorted(telcom_df[x].unique()))

telcom_df = (telcom_df-np.min(telcom_df))/(np.max(telcom_df)-np.min(telcom_df)).values

# Code for splitting the data into train and test split
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


model_rfe=LogisticRegression()
rfe=RFE(model_rfe,1)
rfe_fit=rfe.fit(telcom_df.iloc[:,:32],telcom_df.iloc[:,32])
rfe_fit.n_features_

rfe_fit.ranking_

rank=list(rfe_fit.ranking_)
col_nm=list(telcom_df.iloc[:,:32].columns)
dict_rank={'Column_Name': col_nm,'Ranking':rank}
df_rank=pd.DataFrame(dict_rank)
df_rank.sort_values('Ranking')
X_train, X_test, y_train, y_test = train_test_split(telcom_df[df_rank.loc[df_rank['Ranking']<=25]['Column_Name']],
                                                    telcom_df.iloc[:,32], test_size=0.30,random_state=42)

# Applying K Nearest Neighbour model and finding out the accuracy
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7) #set K neighbor as 3
knn.fit(X_train,y_train)
predicted_y = knn.predict(X_test)
print("KNN accuracy according to K=3 is :",knn.score(X_test,y_test))

# Applying Linear SVM model on the dataset and finding out the accuracy
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
clf = svm.SVC(C=0.5, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovo', degree=3, gamma=1, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
best_model_svm =clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score
print('Accuracy of SVM: ',round(accuracy_score(y_test,yhat)))

# Applying Naive Bayes Classification and finding out the accuracy
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train,y_train)
predicted_y = nb_model.predict(X_test)
accuracy_nb = nb_model.score(X_test,y_test)
print("Naive Bayes accuracy is :",accuracy_nb)