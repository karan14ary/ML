from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


x,y = make_classification(n_samples=1000,n_features=20,n_informative=10,n_redundant=10,n_classes=2,random_state=42)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000,random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy of Logistic Regression model: {accuracy:.2f}')

print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))  
print('\nClassification Report:')
print(classification_report(y_test,y_pred))



