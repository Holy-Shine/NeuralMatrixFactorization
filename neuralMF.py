

import time
import numpy as np
from Dataset import load_rating_data
from keras.models import Model
from keras.layers import InputLayer,Embedding,Flatten,Dense,Merge,Lambda,Input,merge,BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import initializations
from keras import metrics
np.random.seed(0)   # set random seed to recall the best result

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)



def GMF_model(num_users, num_items, latent_dim,regs=[0.0,0.0]):

    # input
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
	
    MF_Embedding_user = Embedding(input_dim = num_users,output_dim = latent_dim, name='user_embedding',
								init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_item = Embedding(input_dim = num_items,output_dim = latent_dim, name='item_embedding',
								init = init_normal, W_regularizer = l2(regs[1]), input_length=1)

	# latent space
    user_latent = Flatten()(MF_Embedding_user(user_input))
    item_latent = Flatten()(MF_Embedding_item(item_input))
	
    predict_vector = merge([user_latent, item_latent], mode = 'mul')
    prediction = Dense(1, activation = None, init = 'lecun_uniform', name = 'prediction')(predict_vector)
    model = Model(input=[user_input,item_input],output=prediction)
    return model

def MLP_model(num_users,num_items,latent_dim,layers=[20,10], reg_layers=[0,0]):
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='item_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')

    # MLP layers
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='sigmoid', name='layer%d' % idx)
        vector = layer(vector)
        vector = BatchNormalization()(vector)

    # BN_vector = BatchNormalization()(vector)
    # Final prediction layer
    prediction = Dense(1, activation=None, init='lecun_uniform', name='prediction')(vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model
if __name__=='__main__':
    num_factors = 10
    regs = [0,0]
    learning_rate = 0.001
    epochs = 100
    batch_size = 128
    trainUser,trainItem, trainLabel, num_users, num_items = load_rating_data("dataset/")
    model = MLP_model(num_users+1,num_items+1,latent_dim=num_factors)
    model.compile(optimizer=Adam(lr=learning_rate),loss='mse')
    model.fit([trainUser,trainItem],trainLabel,batch_size=batch_size,nb_epoch=epochs,verbose=2,validation_split=0.1,shuffle=True)

