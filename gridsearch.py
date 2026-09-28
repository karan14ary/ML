from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['linear','rbf','poly']}
svm__classifier = SVC()
grid = GridSearchCV(svm__classifier,param_grid,refit=True,verbose=2,cv=5,scoring='accuracy')
grid.fit(X_train,y_train)
best_params = grid.best_params_
best_estimator = grid.best_estimator_
y_pred = best_estimator.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print(f"Best Parameters: {best_params}")
print(f"Best Estimator: {best_estimator}")
print(f"Accuracy: {accuracy}")

