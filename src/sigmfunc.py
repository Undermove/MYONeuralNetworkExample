import math

def sigmoid(x):
    for i, element in enumerate(x):
        # if element[1] == 2:
        #     x[i] = [5,5]
        x[i] = 1/(1+math.exp(-element))
    return x