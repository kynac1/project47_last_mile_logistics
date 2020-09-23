import numpy as np
arrivals = np.random.poisson(3, size=5)
print(arrivals)
lt = [[1,2,3]]
max_value = max(lt)
max_ind = lt.index(min(lt))
# max_keys = [k for k, v in lt if v == max_value] # getting all keys containing the `maximum
l=0