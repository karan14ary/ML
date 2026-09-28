import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

x,y =load_digits(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

clf = RandomForestClassifier(random_state=2)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Prediction', fontsize=12)
plt.xlabel('Actual Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()