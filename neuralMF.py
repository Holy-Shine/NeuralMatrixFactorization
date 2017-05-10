

import time
import numpy as np
from Dataset import load_rating_data
from keras.models import Sequential
from keras.layers import InputLayer,Embedding,Flatten,Dense,Merge,Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import initializations
from keras import metrics
np.random.seed(0)   # set random seed to recall the best result

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def myGet_model(num_users, num_items, latent_dim, w_regs=[0,0],o_reg=[10e-6,10e-6], bias_reg=[10e-7,10e-7],glbAvg=3.5):
    model_left,model_right= Sequential(),Sequential()

    # user channel
    model_left.add(InputLayer(input_shape=(1,), input_dtype='float32', name='user_input'))
    model_left.add(Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  init=init_normal, W_regularizer=l2(w_regs[0]), input_length=1))
    model_left.add(Flatten())
    model_left.add(Dense(latent_dim,activation=None,bias=False,activity_regularizer=l2(o_reg[0])))

    # item channel
    model_right.add(InputLayer(input_shape=(1,), input_dtype='float32', name='item_input'))
    model_right.add(Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  init=init_normal, W_regularizer=l2(w_regs[0]), input_length=1))
    model_right.add(Flatten())
    model_right.add(Dense(latent_dim, activation=None,bias=False,activity_regularizer=l2(o_reg[1])))

    model=Sequential()
    model.add(Merge([model_left,model_right],mode='mul'))

    model.add(Dense(1,activation=None,bias=False))
    # user and item bias
    model.add(Dense(1, activation=None, init='uniform', bias=True,b_regularizer=l2(bias_reg[0]),name='user_bias'))
    model.add(Dense(1, activation=None, init='uniform', bias=True, b_regularizer=l2(bias_reg[1]),name='item_bias'))

    # global bias:average rank among data
    model.add(Lambda(lambda x:x+glbAvg))
    return model

if __name__ == '__main__':
    num_factors = 10
    learning_rate = 0.01
    batch_size = 256
	
    t1 = time.time()

    user_input,item_input,labels,num_users,num_items=load_rating_data("dataset/")

    t2 = time.time()
    print "Load data finished : %ss"%(t2-t1)
    model=myGet_model(num_users+1, num_items+1, num_factors)
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse',metrics=[metrics.mse])
    model.fit([user_input, item_input],  # input
               labels,  # labels
               batch_size=batch_size, nb_epoch=100, verbose=2, validation_split=0.1,shuffle=True)