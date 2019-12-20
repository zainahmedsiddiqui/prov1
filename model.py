#Importing libraries
import pandas as pd #For accessing dataset
import numpy as np
import matplotlib.pyplot as plt #For graph.

dataset = pd.read_csv('overallPak.csv') #Reading from folder where csv file is
print(dataset.shape) #Tells how many rows and columns are in the dataset.

#Plotting data points on a graph
#Manual checking if we can find relationship between the data.
#Graph With Matplotlib
#dataset.plot(x='Year', y='Population', style='o')
#plt.title('Pakistan Population')
#plt.xlabel('Year')
#plt.ylabel('Population')
#plt.plot(dataset.Year, dataset.Population, color='red', marker='+')
#plt.show()#The grap shows, there is a linear relation between Year and Population.

#Graph with seaborn
import seaborn as sns
sns.regplot(x="Year", y="Population", data=dataset);


#Preparing the data
X = dataset.iloc[:, :-1].values #Year #Attributes
y = dataset.iloc[:, 1].values #Population #Labels

#split the data 20% test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Linear regression basically finds the best value for the intercept and coefficient,
#which results in a line that best fits the data.
print(regressor.intercept_) 
print(regressor.coef_)

#Making Predictions 
#The y_pred is a numpy array containing 
#predicted values for the input values in the X_test set.
y_pred = regressor.predict(X_test)


#Comparing actual output values for X_test with the predicted values for y_pred
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)


#Evaluation of the Algorithm 
from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:',r2_score(y_test, y_pred))

print(regressor.predict([[2019]]))

#plt.xlabel('Year',fontsize=20)
#plt.ylabel('Population',fontsize=20)
#plt.scatter(dataset.Year,dataset.Population,color='red')
#plt.plot(dataset.Year,regressor.predict(dataset[['Population']]),color='green')

import pickle
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2019]]))

