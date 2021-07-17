import pandas as pd
import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

data = pd.read_csv("C:/Users/venka/StockDataPrediction/StockData.csv")
print(data.head(10))


data.drop(["Date"],axis=1,inplace=True)
data.drop(["OpenInt"],axis=1,inplace=True)
data.drop(["Stock"],axis=1,inplace=True)

print(data.describe())


print(data.info())


#from sklearn.preprocessing import LabelEncoder 
#lab=LabelEncoder()


print(data.isna().any())
print((sns.pairplot(data)))

x=data[['Open','High','Low','Volume']]
y=data['Close']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

print(x_train.shape)
print(x_test.shape)


regressor=LinearRegression()


print(regressor.fit(x_train,y_train))

print(regressor.coef_)
print(regressor.intercept_)


predicated=regressor.predict(x_test)
print(x_test)

print(predicated.shape)

dframe=pd.DataFrame(y_test,predicated)
df=pd.DataFrame({'Actual price':y_test,'Predicated price':predicated})
print(df)

df.head(10)


close=regressor.predict([[0.42388,0.42902,0.41874,23220030]])
print(close)

accuracy=regressor.score(x_test,y_test)
print(accuracy)


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicated))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,predicated))
print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicated)))

graph=df.head(100)
graph.plot(kind='bar')

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
