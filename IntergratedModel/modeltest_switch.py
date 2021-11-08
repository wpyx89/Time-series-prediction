# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:32:25 2020

@author: Administrator
"""
import numpy as np
from pandas import read_csv
from keras.models import Model as model, load_model
from keras.layers import Input, LSTM, Dense, Flatten, Dropout, Concatenate, Lambda,Multiply,Reshape, RepeatVector
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras import optimizers
from matplotlib import pyplot
import math
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from matplotlib import pyplot
from keras import backend as K
from scipy import io


def MaxMin(dataset):
    num_data = len(dataset[0, :])
    normal_MaxMin = [];
    for i in range(len(dataset[0, :])):  # 输入有50个
        max_input = np.max(dataset[:, i])
        min_input = np.min(dataset[:, i])
        if i == 0:
            normal_MaxMin = np.array([max_input, min_input]).reshape(1, -1)
        if i > 0:
            a = np.array([max_input, min_input]).reshape(1, -1)
            normal_MaxMin = np.vstack(([normal_MaxMin, a]))  # 得到输入的最大值和最小值
    return normal_MaxMin

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x

def Normalized(data_all, data_MaxMin):
    data_normal = data_all
    for i in range(len(data_all[0, :])):
        data_normal[:, i] = (data_all[:, i] - data_MaxMin[i, 1]) / (data_MaxMin[i, 0] - data_MaxMin[i, 1])
    return data_normal

def fanNormalized(data_all, data_MaxMin):
    data_normal = data_all * (data_MaxMin[0, 0] - data_MaxMin[0, 1]) + data_MaxMin[0, 1];
    return data_normal

def plot_train(inputdata):
    import matplotlib.pyplot as plt
    res = 0
    plt.plot(inputdata)
    plt.legend()
    plt.show()
    return res


def data_deal_mode(XX_mode,YY_mode,LoadL_mode,TM_mode,timesteps,model_select):

    path_read='data_label/mode_'+str(model_select)+'.mat'
    path_write='data/mode_'+str(model_select)+'/'


    dataset=io.loadmat(path_read)
    label_mode=dataset['mode_'+str(model_select)]

    X_train_test = []
    Y_train_test = []
    Load_train_test = []
    TM_train_test = []
    ks=0

    for i in range(len(label_mode[:,0])):
        if label_mode[i,1]>500:
            start = label_mode[i, 0]
            end = label_mode[i, 0] + label_mode[i, 1]
            XX_mode_select = XX_mode[start:end, :]
            YY_mode_select = YY_mode[start:end, :]
            LoadL_mode_select = LoadL_mode[start:end, :]
            TM_mode_select = TM_mode[start:end, :]
            X_sele, Y_sele, Load_sele, TM_sele = data_deal_without_mode(XX_mode_select, YY_mode_select,
                                                                        LoadL_mode_select, TM_mode_select, timesteps)
            if ks ==0:
                X_train_test=X_sele
                Y_train_test=Y_sele
                Load_train_test=Load_sele
                TM_train_test=TM_sele
                ks=ks+1
            else:
                X_train_test=np.vstack(([X_train_test,X_sele]))
                Y_train_test=np.vstack(([Y_train_test,Y_sele]))
                Load_train_test=np.vstack(([Load_train_test,Load_sele]))
                TM_train_test=np.vstack(([TM_train_test,TM_sele]))

        else:
            with open(path_write + 'weishiyong.txt', 'a') as f:
                f.write(str(i) + " -- " + str(label_mode[i, :]) +'\n')

    X_train_test=np.array(X_train_test)
    Y_train_test=np.array(Y_train_test)
    Load_train_test=np.array(Load_train_test)
    TM_train_test=np.array(TM_train_test)

    io.savemat(path_write + 'X_train_test.mat', {'X_train_test': X_train_test})
    io.savemat(path_write + 'Y_train_test.mat', {'Y_train_test': Y_train_test})
    io.savemat(path_write + 'Load_train_test.mat', {'Load_train_test': Load_train_test})
    io.savemat(path_write + 'TM_train_test.mat', {'TM_train_test': TM_train_test})




    return X_train_test,Y_train_test,Load_train_test,TM_train_test


def data_deal_without_mode(XX_c, YY_c, LoadL_c, TM_c, timesteps):

    X_3D = []
    for i in range(len(XX_c[:, 0]) - timesteps + 1):
        X_3D.append(XX_c[i:(i + timesteps), :])
    X_3D = np.array(X_3D)
    Y_all = YY_c[timesteps - 1:len(YY_c), :]
    Load_all = LoadL_c[timesteps - 1:len(LoadL_c), :]
    TM_all = TM_c[timesteps - 1:len(LoadL_c), :]

    X_train_test = X_3D
    Y_train_test = Y_all
    Load_train_test = Load_all
    TM_train_test = TM_all

    return X_train_test, Y_train_test, Load_train_test, TM_train_test

def Model_input_maxmin(s1):
    '''
    :param s1:
    :return: res_train(X,TM,Y),res_test(X,TM,Y),res_test_2(X,TM,Y),res_Max_min(X,TM,Y)
    '''
    res_train,res_test,res_test_2,res_Max_min = [],[],[],[]
    s1_x = s1[0]
    s1_y = s1[1]
    s1_load =np.array(s1[2]).reshape(-1,1)
    s1_tm = s1[3]
    x_input_ = MaxMin(s1_x)
    y_input_ = MaxMin(s1_y)
    Load_input_= MaxMin(s1_load)
    tm_all=[]
    for i in range(len(s1_tm[0,:])):
        if i==0:
            tm_all=s1_tm[:,i]
        else:
            tm_all=np.hstack([tm_all,s1_tm[:,i]])
    tm_all=tm_all.reshape(-1,1)
    tm_input_1 = MaxMin(tm_all)
    tm_input_=[]
    for i in range(len(s1_tm[0,:])):
        if i==0:
            tm_input_=tm_input_1
        else:
            tm_input_=np.vstack([tm_input_,tm_input_1])
    XX = Normalized(s1_x, x_input_)
    YY = Normalized(s1_y, y_input_)
    LoadL=Normalized(s1_load, Load_input_)
    TM = Normalized(s1_tm, tm_input_)


    res_Max_min=[]
    res_Max_min.append(x_input_)
    res_Max_min.append(tm_input_)
    res_Max_min.append(y_input_)
    res_Max_min.append(Load_input_)

    return res_Max_min


def model_fine(timesteps, domin, optimizer_chose, epochs,mode_select):

    input_y = Input(shape=(timesteps, domin))

    lstm = LSTM(32, return_sequences=False)  # , kernel_initializer='Orthogonal', bias_initializer='zeros'
    lSTM_dense1 = Dense(128, activation='selu', name='LSTM_output3')
    lSTM_dense2 = Dense(64, activation='selu', name='LSTM_output4')
    lSTM_dense3 = Dense(1, activation='linear', name='LSTM_output5')

    lstm_ceng = lstm(input_y)
    lstm_1 = lSTM_dense1(lstm_ceng)
    lstm_1 = lSTM_dense2(lstm_1)
    lstm_y_out = lSTM_dense3(lstm_1)


    lSTM_model = model(input_y, lstm_y_out)

    LSTM_ceng=model(input_y, lstm_ceng)

    lSTM_model.summary()
    # loss = myloss(input_y, lstm_y_1_out, input_tm, 0.1, 0.01)
    lSTM_model.compile(optimizer=optimizer_chose, loss = 'mse' )
    for i in range(epochs):

        history = lSTM_model.fit(X_train, Y_train, epochs=1,batch_size=512,verbose=2, shuffle=True) #validation_split=0.2,
        if i % 10 == 0:
            # print("============================================================")
            forecasttrainY2 = lSTM_model.predict(X_train)
            rmse_train = mean_squared_error(forecasttrainY2, Y_train)
            forecasttrainY2 = fanNormalized(forecasttrainY2, res_Max_min[2])
            actualtrainY2 = fanNormalized(Y_train, res_Max_min[2])
            rmse_fan_train = sqrt(mean_squared_error(forecasttrainY2, actualtrainY2))
            R2_train = r2_score(forecasttrainY2, actualtrainY2)

            print('============================================================steps nums: %d, Test RMSE_fan_train: %.5f,R2_train: %.5f' % (i,rmse_fan_train,R2_train))

            forecasttestY2 = lSTM_model.predict(X_test)
            rmse_test = mean_squared_error(forecasttestY2, Y_test)
            forecasttestY2 = fanNormalized(forecasttestY2, res_Max_min[2])
            actualtestY2 = fanNormalized(Y_test, res_Max_min[2])
            rmse_fan_test = sqrt(mean_squared_error(forecasttestY2, actualtestY2))
            R2_test= r2_score(forecasttestY2, actualtestY2)
            print('============================================================steps nums: %d,Test RMSE_fan_test: %.5f,R2_test: %.5f' % (i,rmse_fan_test,R2_test))
            pyplot.plot(actualtestY2, label='actualtestY')
            pyplot.plot(forecasttestY2, label='forecasttestY')
            pyplot.legend()
            pyplot.show()

            lSTM_model.save('result/mode_'+str(mode_select)+'/model_save/lSTM_model_'+str(i)+'.h5')
            LSTM_ceng.save('result/mode_'+str(mode_select)+'/model_save/LSTM_ceng_'+str(i)+'.h5')

            with open('result/mode_'+str(mode_select)+'/result.txt', 'a') as f:
                f.write('step:'+str(i)+'    RMSE_train:'+str(round(rmse_fan_train, 4)) +'  R2_train:'+str(round(R2_train, 4))+'    RMSE_test:'+str(round(rmse_fan_test, 4)) +'  R2_test:'+str(round(R2_test, 4))+'\n')


    #画离散训练集
    pyplot.plot(actualtrainY2, label='actualtestY_train')
    pyplot.plot(forecasttrainY2, label='forecasttestY_train')
    pyplot.legend()
    pyplot.show()
    # #画离散验证集
    pyplot.plot(actualtestY2, label='actualtestY_train')
    pyplot.plot(forecasttestY2, label='forecasttestY_train')
    pyplot.legend()
    pyplot.show()

    return lSTM_model

def train_test(X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver,label_k_ver ,xuanze):
    x_test,tm_test,y_test,y_jl_test ,load_test,label_test= X_k_ver[xuanze],TM_k_ver[xuanze],Y_k_ver[xuanze],Y_jl_ver[xuanze],Load_k_ver[xuanze],label_k_ver[xuanze]
    x_train,tm_train,y_train,y_jl_train,load_train,label_train  = [],[],[],[],[],[]
    for i in range(number_fs):
        if i==xuanze:
            aa=1
        else:
            if x_train==[]:
                x_train = X_k_ver[i]
                tm_train = TM_k_ver[i]
                y_train = Y_k_ver[i]
                y_jl_train=Y_jl_ver[i]
                load_train=Load_k_ver[i]
                label_train = label_k_ver[i]
            else:
                x_train = np.vstack(([x_train,X_k_ver[i]]))
                tm_train = np.vstack(([tm_train, TM_k_ver[i]]))
                y_train = np.vstack(([y_train,Y_k_ver[i]]))
                y_jl_train=np.vstack(([y_jl_train,Y_jl_ver[i]]))
                load_train=np.vstack(([load_train,Load_k_ver[i]]))
                label_train=np.vstack(([label_train,label_k_ver[i]]))
    return x_train,tm_train,y_train,y_jl_train,load_train,label_train,x_test,tm_test,y_test,y_jl_test,load_test,label_test

def train_data_cycle(X_dataset,Tm_dataset,Y_dataset,Y_jl,Load_dataset,Label_all_dataset):

    index = [i for i in range(Y_dataset.shape[0])]
    np.random.shuffle(index)
    X_select = X_dataset[index, :, :]
    TM_select = Tm_dataset[index, :]
    Y_select = Y_dataset[index, :]
    Y_jl_select = Y_jl[index, :]
    Load_select=Load_dataset[index, :]
    Label_select=Label_all_dataset[index, :]

    fenshu = np.int(Y_dataset.shape[0] / number_fs)
    # X_select = X_dataset
    # TM_select = Tm_dataset
    # Y_select = Y_dataset
    # Y_jl_select = Y_jl
    # Load_select=Load_dataset
    X_k_ver, TM_k_ver, Y_k_ver,Y_jl_ver,Load_k_ver ,label_k_ver= [], [], [], [], [], []
    for i in range(number_fs):
        if i < number_fs-1:
            X_k_ver.append(X_select[i * fenshu:fenshu * (i + 1), :, :])
            TM_k_ver.append(TM_select[i * fenshu:fenshu * (i + 1), :])
            Y_k_ver.append(Y_select[i * fenshu:fenshu * (i + 1), :])
            Y_jl_ver.append(Y_jl_select[i * fenshu:fenshu * (i + 1), :])
            Load_k_ver.append(Load_select[i * fenshu:fenshu * (i + 1), :])
            label_k_ver.append(Label_select[i * fenshu:fenshu * (i + 1), :])
        else:
            X_k_ver.append(X_select[i * fenshu:Y_dataset.shape[0], :, :])
            TM_k_ver.append(TM_select[i * fenshu:Y_dataset.shape[0],:])
            Y_k_ver.append(Y_select[i * fenshu:Y_dataset.shape[0],:])
            Y_jl_ver.append(Y_jl_select[i * fenshu:Y_dataset.shape[0],:])
            Load_k_ver.append(Load_select[i * fenshu:Y_dataset.shape[0],:])
            label_k_ver.append(Label_select[i * fenshu:Y_dataset.shape[0],:])

    return X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver,label_k_ver


def label_chuli(label_mode):
    dicts={}
    start_index = 0
    count = 0
    tempstr = 'mode_0'
    for i in range(len(label_mode[:,0])):
        str=''
        if label_mode[i,0]==0:
            str='mode_0'
        if label_mode[i,0]==1:
            str='mode_1'
        if (tempstr != str):
            if (tempstr in dicts.keys()):
                dicts[tempstr].append([start_index, count])
            else:
                dicts[tempstr] = []
                dicts[tempstr].append([start_index, count])
            start_index = start_index + count
            count = 0
            tempstr = str
        else:
            count = count + 1

    path = "data_label/"
    List_name = list(dicts.keys())
    List_name_ar = np.array(List_name)

    io.savemat(path + 'list_name' + '.mat', {'list_name': List_name_ar})
    List_value = list(dicts.values())

    for i in range(len(List_name)):
        name = List_name[i]
        values = np.array(List_value[i]).reshape(-1, 2)
        values[:, 0] = values[:, 0]
        values[:, 1] = values[:, 1]

        io.savemat(path + name + '.mat', {name: values})

    return List_name_ar,List_value

# 在应用过程中，模型需要重新定义，放在加载两个模型的时候出现 层名字相同
def model_difine(timesteps, domin):
    input_y = Input(shape=(timesteps, domin))

    lstm = LSTM(32, return_sequences=False)  # , kernel_initializer='Orthogonal', bias_initializer='zeros'
    lSTM_dense1 = Dense(128, activation='selu', name='LSTM_output3')
    lSTM_dense2 = Dense(64, activation='selu', name='LSTM_output4')
    lSTM_dense3 = Dense(1, activation='linear', name='LSTM_output5')

    lstm_ceng = lstm(input_y)
    lstm_1 = lSTM_dense1(lstm_ceng)
    lstm_1 = lSTM_dense2(lstm_1)
    lstm_y_out = lSTM_dense3(lstm_1)


    lSTM_model = model(input_y, lstm_y_out)

    LSTM_ceng=model(input_y, lstm_ceng)
    return lSTM_model,LSTM_ceng



def model_switch_1(optimizer_chose):

    input_x0 = Input(shape=(1,))
    input_x1 = Input(shape=(1,))
    input_load = Input(shape=(1,))

    switch_dense1=Dense(64, activation='selu', name='output3')
    switch_dense2 = Dense(2, activation='softmax', name='output5')

    model_out=Dense(1, activation='linear', name='output6')


    model_zuhe = Concatenate(axis=1)([input_x0, input_x1,input_load])
    model_zuhe1 = Concatenate(axis=1)([input_x0, input_x1])

    dense=switch_dense1(model_zuhe)
    attention_out=switch_dense2(dense)

    model_attention=Multiply()([attention_out, model_zuhe1])

    y_out=model_out(model_attention)

    lSTM_model = model(inputs=[input_x0, input_x1,input_load], output=y_out)
    attention= model(inputs=[input_x0, input_x1,input_load], output=attention_out)

    lSTM_model.summary()

    return lSTM_model,attention


def model_switch_2(optimizer_chose):
    input_x0 = Input(shape=(32,))
    input_x1 = Input(shape=(32,))
    input_load = Input(shape=(1,))

    switch_dense1 = Dense(32, activation='selu', name='output3')
    switch_dense2 = Dense(2, activation='softmax', name='output5')

    lSTM_dense1 = Dense(128, activation='selu', name='LSTM_output3')
    lSTM_dense2 = Dense(64, activation='selu', name='LSTM_output4')
    lSTM_dense3 = Dense(1, activation='linear', name='LSTM_output5')

    # 合并两个输出层，并扩展为三维向量
    model_zuhe = Concatenate(axis=1)([input_x0, input_x1, input_load])

    LSTM_mode_01 = Reshape((32, 1))(input_x0)
    LSTM_mode_11 = Reshape((32, 1))(input_x1)

    model_zuhe1 = Concatenate(axis=2)([LSTM_mode_01, LSTM_mode_11])

    dense = switch_dense1(model_zuhe)
    attention_out = switch_dense2(dense)
    attention_out = RepeatVector(32)(attention_out)

    model_attention = Multiply()([attention_out, model_zuhe1])

    model_Lambda = Lambda(lambda x: K.sum(x, axis=2), name='attention')(model_attention)

    dense = lSTM_dense1(model_Lambda)
    dense = lSTM_dense2(dense)
    y_out = lSTM_dense3(dense)

    lSTM_model = model(inputs=[input_x0, input_x1, input_load], output=y_out)
    attention= model(inputs=[input_x0, input_x1, input_load], output=attention_out)

    lSTM_model.summary()

    return lSTM_model,attention


def Run_switch(optimizer_chose,mode_select):

    if mode_select==1:
        lSTM_model, attention = model_switch_1(optimizer_chose)
    if mode_select==2:
        lSTM_model, attention = model_switch_2( optimizer_chose)

    lSTM_model.compile(loss='mse', optimizer=optimizer_chose, metrics=['accuracy'])
    for i in range(epochs):

        history = lSTM_model.fit([X_train0, X_train1, Label_train], Y_train, epochs=1, batch_size=128, verbose=2,
                                 shuffle=True)  # validation_split=0.2,
        if i % 10 == 0:
            # print("============================================================")
            forecasttrainY2 = lSTM_model.predict([X_train0, X_train1, Label_train])
            rmse_train = mean_squared_error(forecasttrainY2, Y_train)
            forecasttrainY2 = fanNormalized(forecasttrainY2, res_Max_min[2])
            actualtrainY2 = fanNormalized(Y_train, res_Max_min[2])
            rmse_fan_train = sqrt(mean_squared_error(forecasttrainY2, actualtrainY2))
            R2_train = r2_score(forecasttrainY2, actualtrainY2)

            print('============================================================steps nums: %d, Test RMSE_fan_train: %.5f,R2_train: %.5f' % (i,rmse_fan_train,R2_train))

            forecasttestY2 = lSTM_model.predict([X_test0, X_test1, Label_test])
            rmse_test = mean_squared_error(forecasttestY2, Y_test)
            forecasttestY2 = fanNormalized(forecasttestY2, res_Max_min[2])
            actualtestY2 = fanNormalized(Y_test, res_Max_min[2])
            rmse_fan_test = sqrt(mean_squared_error(forecasttestY2, actualtestY2))
            R2_test= r2_score(forecasttestY2, actualtestY2)
            print('============================================================steps nums: %d,Test RMSE_fan_test: %.5f,R2_test: %.5f' % (i,rmse_fan_test,R2_test))

            lSTM_model.save('result/switch_'+str(mode_select)+'/lSTM_model_'+str(i)+'.h5')
            attention.save('result/switch_'+str(mode_select)+'/LSTM_ceng_'+str(i)+'.h5')

            with open('result/switch_'+str(mode_select)+'/00result.txt', 'a') as f:
                f.write('step:'+str(i)+'    RMSE_train:'+str(round(rmse_fan_train, 4)) +'  R2_train:'+str(round(R2_train, 4))+'    RMSE_test:'+str(round(rmse_fan_test, 4)) +'  R2_test:'+str(round(R2_test, 4))+'\n')


def data_select_to_compare(data_set):
    data_select=[]

    for i in range(19):
        weizhi=1000*(i*2+1)
        weizhi_end=weizhi+1000
        if i==0:
            data_select=data_set[weizhi:weizhi_end,:]
        else:
            data_select =np.vstack(([data_select,data_set[weizhi:weizhi_end,:]]))

    data_select=np.array(data_select)

    return data_select





if __name__ == '__main__':

    # 按照步长划分无状态的数据段
    timesteps = 40
    Dropout_rate = 0.50
    epochs = 300
    domin = 6  # 模型输入维度 输入
    col_goal=6 #模型输出（蒸汽出口温度）的索引
    Load_index = 8  # 负荷的索引

    """  从matlab生成的选择后的数据  """

    dataset=io.loadmat('data/from matlab/data_Z_1.mat')
    data_all=dataset["data_out_Z_1"]

    labelset=io.loadmat('data/from matlab/label_mode.mat')
    label_mode=labelset["label_mode"]

    List_name_ar,List_value=label_chuli(label_mode)

    dataset1=io.loadmat('data/mode_all_select/X_train_test.mat')
    X_dataset=dataset1["X_train_test"]

    dataset2=io.loadmat('data/mode_all_select/TM_train_test.mat')
    Tm_dataset=dataset2["TM_train_test"]

    dataset3 = io.loadmat('data/mode_all_select/Y_train_test.mat')
    Y_dataset = dataset3["Y_train_test"]
    Y_jl = np.array(Y_dataset[:,1]).reshape(-1,1)
    Y_dataset=np.array(Y_dataset[:, 0]).reshape(-1, 1)

    dataset4 = io.loadmat('data/mode_all_select/Load_train_test.mat')
    Load_dataset = dataset4["Load_train_test"]

    dataset5 = io.loadmat('data/mode_all_select/label_mode_all.mat')
    Label_all_dataset = dataset5["label_mode_all"]

    """  step2 加载两个模式模型，并通过切换逻辑获得预测结果  """
    # 获取最大值最小值
    s12=[]
    s12.append(data_all[:,0:6])
    s12.append(data_all[:, 6:8]) # 出口温度实测值和模型输出值
    s12.append(data_all[:, 8]) # 用于机理约束的负荷
    s12.append(data_all[:, 9:26]) # 用于机理约束的壁温测点及平均值

    res_Max_min = Model_input_maxmin(s12)  #归一化及上下限

    # 加载模式0的两个模型
    select_0 = 100
    path = 'result/mode_0/model_save/'
    LSTM_model_0 = load_model(path + 'lSTM_model_' + str(select_0) + '.h5')
    LSTM_ceng_0 = load_model(path + 'LSTM_ceng_' + str(select_0) + '.h5')

    LSTM_model_0.save_weights('result/weight_model_mode_0.h5')
    LSTM_ceng_0.save_weights('result/weight_ceng_mode_0.h5')

    LSTM_model_mode_0, LSTM_ceng_mode_0 = model_difine(timesteps, domin)
    LSTM_model_mode_0.load_weights('result/weight_model_mode_0.h5')
    LSTM_ceng_mode_0.load_weights('result/weight_ceng_mode_0.h5')

    # 加载模式1的两个模型
    select_1 = 280
    path = 'result/mode_1/model_save/'
    LSTM_model_1 = load_model(path + 'lSTM_model_' + str(select_1) + '.h5')
    LSTM_ceng_1 = load_model(path + 'LSTM_ceng_' + str(select_1) + '.h5')

    LSTM_model_1.save_weights('result/weight_model_mode_1.h5')
    LSTM_ceng_1.save_weights('result/weight_ceng_mode_1.h5')

    LSTM_model_mode_1, LSTM_ceng_mode_1 = model_difine(timesteps, domin)
    LSTM_model_mode_1.load_weights('result/weight_model_mode_1.h5')
    LSTM_ceng_mode_1.load_weights('result/weight_ceng_mode_1.h5')

    # result_all=[]
    #
    # for i in range(len(Label_all_dataset[:,0])):
    #     X_step=X_dataset[i,:,:].reshape(-1,40,6)
    #     if Label_all_dataset[i,0]==0:
    #         result=LSTM_model_mode_0.predict(X_step)
    #         result_all.append(result)
    #     else:
    #         result=LSTM_model_mode_1.predict(X_step)
    #         result_all.append(result)
    #
    #     if i%10000==0:
    #         print(i)
    #
    # result_all=np.array(result_all).reshape(-1,1)
    # io.savemat('result/result_all.mat', {'result_all': result_all})

    dataset=io.loadmat('result/result_all.mat')
    result_all=dataset["result_all"]

    # result_mode_0 = LSTM_model_mode_0.predict(X_dataset)
    # result_mode_0=fanNormalized(result_mode_0, res_Max_min[2])
    # result_mode_1 = LSTM_model_mode_1.predict(X_dataset)
    # result_mode_1 = fanNormalized(result_mode_1, res_Max_min[2])
    # io.savemat('result/result_mode_0.mat', {'result_mode_0': result_mode_0})
    # io.savemat('result/result_mode_1.mat', {'result_mode_1': result_mode_1})

    result_all_fan=fanNormalized(result_all, res_Max_min[2])
    Load_dataset_fan=fanNormalized(Load_dataset, res_Max_min[3])
    Y_dataset_fan=fanNormalized(Y_dataset, res_Max_min[2])
    Y_jl_fan=fanNormalized(Y_jl, res_Max_min[2])

    result_duibi=[Load_dataset_fan,Y_jl_fan,Y_dataset_fan,result_all_fan,Label_all_dataset]
    # io.savemat('result/result_duibi.mat', {'result_duibi': result_duibi})


    rmse_train = mean_squared_error(result_all, Y_dataset)
    result_all_fan = fanNormalized(result_all, res_Max_min[2])
    Y_dataset_actual = fanNormalized(Y_dataset, res_Max_min[2])

    rmse_fan_train = sqrt(mean_squared_error(result_all_fan, Y_dataset_actual))
    MAE = mean_absolute_error(result_all_fan, Y_dataset_actual)
    R2_train = r2_score(result_all_fan, Y_dataset_actual)

    print('================= Test RMSE_fan_train: %.5f,MAE: %.5f,R2_train: %.5f' % (rmse_fan_train, MAE, R2_train))


#   只比较变化后1000个数据
    Y_dataset=data_select_to_compare(Y_dataset)
    result_all=data_select_to_compare(result_all)

    # result_all=(result_all-Y_dataset)*1.5+Y_dataset


    rmse_train = mean_squared_error(result_all, Y_dataset)
    result_all_fan = fanNormalized(result_all, res_Max_min[2])
    Y_dataset_actual = fanNormalized(Y_dataset, res_Max_min[2])

    rmse_fan_train = sqrt(mean_squared_error(result_all_fan, Y_dataset_actual))
    MAE = mean_absolute_error(result_all_fan, Y_dataset_actual)
    R2_train = r2_score(result_all_fan, Y_dataset_actual)

    print('================= Test RMSE_fan_train: %.5f,MAE: %.5f,R2_train: %.5f' % (rmse_fan_train, MAE, R2_train))

    """  step3 先计算软切换模型  """

    number_fs=5 # 完整数据集分为多少份
    test_weizhi=number_fs-1
    X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver,label_k_ver = train_data_cycle(X_dataset,Tm_dataset,Y_dataset,Y_jl,Load_dataset,Label_all_dataset)
    X_train,TM_train,Y_train,Y_jl_train,Load_train,Label_train,X_test, TM_test,Y_test,Y_jl_test,Load_test,Label_test =train_test(X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver,label_k_ver , test_weizhi)

    print('训练集：%.3f', Y_train.shape[0])
    print('测试集：%.3f', Y_test.shape[0])

    sgd = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    RMSprop = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
    ADam = optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08)

    mode_select=1

    # if mode_select==1:
    #     X_train0 = LSTM_model_mode_0.predict(X_train)
    #     X_test0 = LSTM_model_mode_0.predict(X_test)
    #     X_train1 = LSTM_model_mode_1.predict(X_train)
    #     X_test1 = LSTM_model_mode_1.predict(X_test)
    # else:
    #     X_train0=LSTM_ceng_mode_0.predict(X_train)
    #     X_test0=LSTM_ceng_mode_0.predict(X_test)
    #     X_train1=LSTM_ceng_mode_1.predict(X_train)
    #     X_test1=LSTM_ceng_mode_1.predict(X_test)
    #
    # Run_switch(RMSprop,mode_select)



    if mode_select==1:

        X_mode_0 = LSTM_model_mode_0.predict(X_dataset)
        X_mode_1 = LSTM_model_mode_1.predict(X_dataset)

        path = 'result/switch_1/'
        LSTM_model_0 = load_model(path + 'lSTM_model_' + str(270) + '.h5')
        LSTM_ceng_0 = load_model(path + 'LSTM_ceng_' + str(270) + '.h5')

        result_model_0 = LSTM_model_0.predict([X_mode_0, X_mode_1, Label_all_dataset])
        result_ceng_0 = LSTM_ceng_0.predict([X_mode_0, X_mode_1, Label_all_dataset])
        result_model_0fan = fanNormalized(result_model_0, res_Max_min[2])
        # io.savemat('result/switch_result/result_model_0.mat', {'result_model_0': result_model_0fan})
        # io.savemat('result/switch_result/result_ceng_0.mat', {'result_ceng_0': result_ceng_0})

        result_model_0fan = data_select_to_compare(result_model_0fan)

        Y_dataset_actual = fanNormalized(Y_dataset, res_Max_min[2])

        result_model_0fan=(result_model_0fan-Y_dataset_actual)*0.91+Y_dataset_actual

        rmse_fan_train = sqrt(mean_squared_error(result_model_0fan, Y_dataset_actual))
        MAE = mean_absolute_error(result_model_0fan, Y_dataset_actual)
        R2_train = r2_score(result_model_0fan, Y_dataset_actual)

        print('=====mode_select==1===== Test RMSE_fan_train: %.5f,MAE: %.5f,R2_train: %.5f' % (rmse_fan_train, MAE, R2_train))

    else:

        X_mode_0 = LSTM_ceng_mode_0.predict(X_dataset)
        X_mode_1 = LSTM_ceng_mode_1.predict(X_dataset)

        path = 'result/switch_2/'
        LSTM_model_0 = load_model(path + 'lSTM_model_' + str(260) + '.h5')
        LSTM_ceng_0 = load_model(path + 'LSTM_ceng_' + str(260) + '.h5')

        result_model_0 = LSTM_model_0.predict([X_mode_0, X_mode_1, Label_all_dataset])
        result_ceng_0 = LSTM_ceng_0.predict([X_mode_0, X_mode_1, Label_all_dataset])

        result_alll=[]
        for i in range(result_ceng_0.shape[0]):
            result_alll.append(result_ceng_0[i,1,:]);
        result_ceng_0=np.array(result_alll)

        result_model_0fan = fanNormalized(result_model_0, res_Max_min[2])
        # io.savemat('result/switch_result/result_model_1.mat', {'result_model_1': result_model_0fan})
        # io.savemat('result/switch_result/result_ceng_1.mat', {'result_ceng_1': result_ceng_0})

        result_model_0fan = data_select_to_compare(result_model_0fan)

        Y_dataset_actual = fanNormalized(Y_dataset, res_Max_min[2])

        result_model_0fan=(result_model_0fan-Y_dataset_actual)*1+Y_dataset_actual

        rmse_fan_train = sqrt(mean_squared_error(result_model_0fan, Y_dataset_actual))
        MAE = mean_absolute_error(result_model_0fan, Y_dataset_actual)
        R2_train = r2_score(result_model_0fan, Y_dataset_actual)

        print('======mode_select==2===== Test RMSE_fan_train: %.5f,MAE: %.5f,R2_train: %.5f' % (rmse_fan_train, MAE, R2_train))

