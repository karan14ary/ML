import numpy as np

def calculate_percentile(data,percentile):
    return np.percentile(data,percentile)

data_set_1 =[10,15,20,25,30]
percentile_result_1= calculate_percentile(data_set_1,75)
print("75 Percentile: ",percentile_result_1)
data_set_2 =[5,10,10,15,20,25,30,35]
percentile_result_2= calculate_percentile(data_set_2,50)
print("50th Percentile: ",percentile_result_2)