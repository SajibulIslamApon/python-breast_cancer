import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data=load_breast_cancer()


#print(data)
#print(data.feature_names)
#print(data.target_names)


data_frame=pd.DataFrame(np.c_[data.data,data.target],columns=[list(data.feature_names)+['target']])


#print(data_frame.head())
#print(data_frame.tail())
#print(data_frame.shape)


x=data_frame.iloc[:,0:-1]
y=data_frame.iloc[:,-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#,test_size=0.2,random_state=2020
#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)




scaler = StandardScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)



classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
classifier.score(x_test,y_test)
print(classifier.score(X_test,y_test))

y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


