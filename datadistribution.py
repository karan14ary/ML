import numpy as np
import matplotlib.pyplot as plt

datadistribution =np.random.normal(loc=50,scale=10,size=1000)

plt.hist(datadistribution,bins=30,edgecolor='black')
plt.title('Histrogram - Data Distribution')
plt.xlabel('Values')
plt.ylabel("Frequency")
plt.show()
