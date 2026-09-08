from tensorflow import keras
from tensorflow.keras import layers
# -*- coding: UTF-8 -*-
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input,AveragePooling1D,MaxPool1D,Convolution1D,MaxPool2D,GlobalAveragePooling2D,AveragePooling2D,Embedding,Bidirectional,MultiHeadAttention
from tensorflow.python.keras.layers import BatchNormalization, Dropout, Flatten, Dense,Concatenate,GlobalAveragePooling1D, Reshape,Multiply,GlobalMaxPooling1D,GRU,LSTM
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model
import scipy as sp
from scipy.stats import pearsonr
from tensorflow.python.keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Multiply
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import *
import pandas as pd
import numpy as np
import time

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

#one-hot encoding
def grna_preprocess1(lines, length):
    data_n = len(lines)
    seq = np.zeros((data_n,1, length, 4), dtype=int)
    for l in range(data_n):
        data = lines[l]
        seq_temp = data
        for i in range(length):
            if seq_temp[i] in "Aa":
                seq[l, 0, i, 0] = 1
            elif seq_temp[i] in "Cc":
                seq[l, 0, i, 1] = 1
            elif seq_temp[i] in "Gg":
                seq[l, 0, i, 2] = 1
            elif seq_temp[i] in "Tt":
                seq[l, 0, i, 3] = 1
    return seq

#position encoding
def position_code(lines,length):
    data_n=len(lines)
    print(data_n)
    seq = np.zeros((data_n, 1, length, 1), dtype=int)
    for l in range(data_n):
        data = lines[l]
        for i in range(length):
            if data[i] in 'Aa':
                seq[l, 0, i, 0] = i+1
            elif data[i] in "Cc":
                seq[l, 0, i, 0] = i+22
            elif data[i] in "Gg":
                seq[l, 0, i, 0] = i+43
            elif data[i] in "Tt":
                seq[l, 0, i, 0] = i+64
    return seq
#read csv_file
print("Data downloading and pre-processing ... ")
FILE = pd.read_csv(r'D:\eSpCas 9.csv')

data = pd.DataFrame(columns=(['21mer']))
data['21mer'] = FILE['21mer']
x_data = data.iloc[:, 0]

x_data1 = grna_preprocess1(x_data,21)
x_data2=position_code(x_data,21)


y_data =FILE['ESp_Efficiency']
y_data = np.array(y_data)
y_data = y_data.reshape(len(y_data), -1)

#read DNA shape feature
def get_shapes(shapes,shape_path):
    shape_series = []
    for shape in shapes:
        shape_series.append(pd.read_csv(shape_path + '/' + 'esp' + '_' + shape + '.csv'))
    completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))
    for i in range(len(shapes)):
        shape_samples = shape_series[i]
        for m in range(shape_samples.shape[0]):
            completed_shape[m][i] = shape_samples.loc[m]
    completed_shape = np.transpose(completed_shape, [0, 2, 1])
    return completed_shape
Wt_shape=get_shapes(shapes=['EP','HeIT','MGW','ProT','Roll'],shape_path=r'D:\DNA_shape\esp_dataset')

#read hand-crafted biological feature
bio_feature = pd.read_csv(r'D:\esp_biofeature.csv',header=None)
bio_feature = np.array(bio_feature)
#read secondary structure feature
x_data_RNA_Struct = pd.read_csv(r'D:\eSpCas9_RNAencoding.csv',header=None)
x_data_RNA_Struct = np.array(x_data_RNA_Struct)
x_data_RNA_Struct = np.expand_dims(x_data_RNA_Struct, axis=-1)
x_data_RNA_Struct = np.expand_dims(x_data_RNA_Struct, axis=1)

#one-hot encoding and secondary structure feature
model_input1 = np.concatenate((x_data1,x_data_RNA_Struct), axis=3)
x_train_posi,x_test_posi,x_train_bio,x_test_bio,x_train_oneRNA,x_test_oneRNA,x_train_one,x_test_one, x_train_shape,x_test_shape,y_train, y_test = train_test_split(x_data2,bio_feature_add,model_input1,x_data1,f,y_data, test_size=0.15,random_state=1)

x_train_one=x_train_one.reshape(49824,1*21,4)
x_test_one=x_test_one.reshape(8793,1*21,4)

x_train_shape=x_train_shape.reshape(49824,1*21,5)
x_test_shape=x_test_shape.reshape(8793,1*21,5)

x_train_oneRNA=x_train_oneRNA.reshape(49824,1*21,5)
x_test_oneRNA=x_test_oneRNA.reshape(8793,1*21,5)

x_train_posi=x_train_posi.reshape(49824,1*21*1)
x_test_posi=x_test_posi.reshape(8793,1*21*1)


def cnn_block1(input):
    data_Conv1 = Convolution1D(filters=30, kernel_size=1, padding='same', activation='relu')(input)
    data_Conv2 = Convolution1D(filters=30, kernel_size=2, padding='same', activation='relu')(input)
    data_Conv3 = Convolution1D(filters=30, kernel_size=3, padding='same', activation='relu')(input)
    data_Conv4 = Convolution1D(filters=30, kernel_size=4, padding='same', activation='relu')(input)
    data_Conv5 = Convolution1D(filters=30, kernel_size=5, padding='same', activation='relu')(input)
    data_t = Concatenate()([data_Conv1, data_Conv2, data_Conv3, data_Conv4, data_Conv5])
    X1 = MaxPool1D(strides=2, padding='same')(data_t)
    X2 = AveragePooling1D(strides=2, padding='same')(data_t)
    data_r = Concatenate()([X1, X2])
    out = Dropout(0.2)(data_r)
    out1 = Flatten()(out)
    return out1

def cnn_block2(input):
    data_Conv1 = Convolution1D(filters=30, kernel_size=1, padding='same', activation='relu')(input)
    data_Conv3 = Convolution1D(filters=30, kernel_size=4, padding='same', activation='relu')(data_Conv1)
    data_Conv5 = Convolution1D(filters=30, kernel_size=5, padding='same', activation='relu')(data_Conv3)
    out = BatchNormalization()(data_Conv5)
    out = MaxPool1D(strides=2, padding='same')(out)
    out = Dropout(0.25)(out)
    channel = int(out.shape[-1])
    print('channel',channel)
    x = GlobalAveragePooling1D()(out)
    x = Dense(int(channel / 4), activation='hard_sigmoid')(x)
    x = Dense(channel, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, channel))(x)
    x = Multiply()([out, x])
    out2 = Flatten()(x)
    return out2

SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = True
INPUT_DIM = 2
TIME_STEPS = 21
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
earlyStop = [EarlyStopping(monitor='val_loss',patience=30, mode='min', verbose=1, restore_best_weights = True)]



def DeepMEns_esp(shape1, shape2, shape3, shape4):
    kf = KFold(n_splits=5, shuffle=True, random_state=32)
    print('Model building ... ')
    N = 0
    for train_index, valid_index in kf.split(x_train_one):
        train_shape, valid_shape = x_train_shape[train_index], x_train_shape[valid_index]
        train_oneRNA, valid_oneRNA = x_train_oneRNA[train_index], x_train_oneRNA[valid_index]
        train_posi, valid_posi = x_train_posi[train_index], x_train_posi[valid_index]
        train_bio, valid_bio = x_train_bio[train_index], x_train_bio[valid_index]
        train_y, valid_y = y_train[train_index], y_train[valid_index]
        print('Model building ... ')

        DNAshape_input = Input(shape=shape1, name='shape')
        data_input2 = TransformerEncoder(embed_dim=5, dense_dim=64, num_heads=6)(DNAshape_input)
        out_shape = cnn_block2(data_input2)

        oneRNA_input = Input(shape=shape2, name='sequenceRNA')
        data_input1 = TransformerEncoder(embed_dim=5, dense_dim=64, num_heads=6)(oneRNA_input)
        outRNA_one = cnn_block1(data_input1)

        posi_input = Input(shape=shape3, name='posi')
        embedding_layer = Embedding(input_dim=1000, output_dim=4, input_length=21, embeddings_initializer='uniform')
        embedding = embedding_layer(posi_input)
        attention_mul = attention_3d_block(embedding)
        lstm_units = 64
        attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)

        biological_input = Input(shape=shape4, name='bio_input')

        con = Concatenate()([out_shape, outRNA_one, attention_mul, biological_input])

        BN1 = BatchNormalization()(con)
        f1 = Dense(320, activation='relu')(BN1)
        BN2 = BatchNormalization()(f1)
        drop1 = Dropout(0.6)(BN2)
        f2 = Dense(240, activation='relu')(drop1)
        BN3 = BatchNormalization()(f2)
        drop2 = Dropout(0.4)(BN3)
        f3 = Dense(160, activation='relu')(drop2)
        BN4 = BatchNormalization()(f3)
        drop3 = Dropout(0.2)(BN4)
        output = Dense(1, activation="hard_sigmoid", name="output")(drop3)
        model = Model(inputs=[DNAshape_input, oneRNA_input, posi_input, biological_input], outputs=[output])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss='mse')
        model.fit([train_shape, train_oneRNA, train_posi, train_bio], train_y, batch_size=500, epochs=300,
                  validation_data=([valid_shape, valid_oneRNA, valid_posi, valid_bio], valid_y), callbacks=earlyStop)
        model.save(r'D:\esp_DeepMEns_{0}.h5'.format(N))
        model.summary()
        N = N + 1
f = model_4(shape1=(21, 5), shape2=(21, 5), shape3=(21,), shape4=(17,))



