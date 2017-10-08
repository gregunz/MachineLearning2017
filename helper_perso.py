import numpy as np

def range_feature(f, data):
    d = data[1][data[1][:,f] != -999]
    print("Feature {f} range from {min} to {max}".format(
              f=f, min=np.min(d[:,f]), max=np.max(d[:,f])))
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    index = np.random.permutation(len(y))
    x_rand = x[index]
    y_rand = y[index]
    split = int(np.floor(len(y) * ratio))
    return (x_rand[:split], y_rand[:split], x_rand[split:], y_rand[split:])
