from sklearn.model_selection import train_test_split
import numpy as np



np.random.seed(42)
x=np.random.rand(100,1)
y=2* x.squeeze()+1+0.1*np.random.randn(100)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(f"Training ser size: {len(x_train)}")
print(f"Testing ser size: {len(x_test)}")
print(f"Training ser size: {len(y_train)}")
print(f"Training ser size: {len(y_train)}")