from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data= pd.read_csv('loser2.csv')
print(data.columns)

#checking for attributes with null values
print(data.isnull().sum())

#Finding and displaying correlation between attributes
correlation = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

#Dropping columns having less correlation
data=data.drop(columns=['free sulfur dioxide','fixed acidity','volatile acidity','chlorides','total sulfur dioxide','density'],axis=1)


#Splitting data
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#creation of regression model and training it
model=LinearRegression().fit(X_train,y_train)

#predicting the target
predict=model.predict(X_test)

#evaluation of model using metrics
mean_squared_error = mean_squared_error(y_test, predict)
r2_score = r2_score(y_test,predict)
print("Mean squared error :",mean_squared_error)
print("R2 score : ",r2_score)
