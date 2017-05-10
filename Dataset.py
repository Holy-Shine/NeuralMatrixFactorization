import scipy.sparse as sp
import numpy as np
import cPickle
np.random.seed(0)

def load_rating_data(filepath):
    filepath += "rating-10k.dat"
    dataList=[]
    last_user=0
    num_users, num_items = 0,0
    with open(filepath,'r') as f:
        for line in f:
            arr = line.split("::")
            if len(arr)>2:
                user, item, rank = int(arr[0]), int(arr[1]), float(arr[2])
                num_users=max(user,num_users)
                num_items=max(num_items,item)
                dataList.append([user, item, rank])
    np.random.shuffle(dataList) # shuffle the dataSet
    dataArray = np.array(dataList,dtype="float32")
    return dataArray[:,0],dataArray[:,1],dataArray[:,2],num_users,num_items