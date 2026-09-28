from sklearn.model_selection import cross_val_score,KFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target

svm = SVC(kernel='linear', C=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm, X, y, cv=kfold)
print(f'Cross-Validation Accuracy Scores: {accuracy_scores}')
print(f'Mean Accuracy: {accuracy_scores.mean():.2f}')

