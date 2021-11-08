# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:32:25 2020

@author: Administrator
"""
import numpy as np
from pandas import read_csv
from keras.models import Model as model, load_model
from keras.layers import Input, LSTM, Dense, Flatten, Dropout, Concatenate, Lambda,SimpleRNN,GRU
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras import optimizers
from matplotlib import pyplot
import math
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot
from keras import backend as K
from scipy import io
import time

def write_mat(data,path,name):
    data_zh=[]
    for i in range (len(data)):
        if i==0:
            data_zh=np.array(data[i])
        else:
            data_zh=np.hstack(([data_zh,np.array(data[i])]))

    io.savemat(str(path)+str(name)+'.mat', {'data': data_zh})
    return 1

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

def Model_input(s1, length_use, timesteps, domin):
    '''
    :param s1:
    :return: res_train(X,TM,Y),res_test(X,TM,Y),res_test_2(X,TM,Y),res_Max_min(X,TM,Y)
    '''
    res_train,res_test,res_test_2,res_Max_min = [],[],[],[]
    s1_x = s1[0]
    s1_y = s1[1]
    s1_load =np.array(s1[2]).reshape(-1,1)
    s1_tm = s1[3]
    length_use_v = int(length_use / timesteps)
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


    X_3D=[]
    for i in range(len(XX[:,0])-timesteps+1):
        X_3D.append(XX[i:(i+timesteps),:])
    X_3D=np.array(X_3D)
    Y_all=YY[timesteps-1:len(YY),:]
    Load_all=LoadL[timesteps-1:len(LoadL),:]
    TM_all=TM[timesteps-1:len(LoadL),:]

    X_train_test=X_3D[0:length_use,:,:]
    Y_train_test= Y_all[0:length_use, :]
    Load_train_test= Load_all[0:length_use, :]
    TM_train_test = TM_all[0:length_use, :]

    X_test_fh=X_3D[length_use:len(Y_all[:,0]),:,:]
    Y_test_fh= Y_all[length_use:len(Y_all[:,0]), :]
    Load_test_fh= Load_all[length_use:len(Y_all[:,0]), :]
    TM_test_fh = TM_all[length_use:len(Y_all[:,0]), :]

    res_Max_min=[]
    res_Max_min.append(x_input_)
    res_Max_min.append(tm_input_)
    res_Max_min.append(y_input_)
    res_Max_min.append(Load_input_)

    return X_train_test,Y_train_test,Load_train_test,TM_train_test,X_test_fh,Y_test_fh,Load_test_fh,TM_test_fh,res_Max_min

def Model_cycle(model,epochs,nums):
    '''
    :param model: 输入模型
    :param epochs: 更新迭代次数
    :param nums: 记录总体模型迭代次数，当达到10次保存
    :return: 返回新模型
    '''
    for i in range(epochs):
        history = model.fit([X_train,TM_train], Y_train, epochs=1,  batch_size=128,
                                  verbose=0, shuffle=True) #validation_split=0.2,
        if i % 5 == 0:
            print("============================================================")
            forecasttrainY2 = model.predict([X_train,TM_train])
            rmse_train = mean_squared_error(forecasttrainY2, Y_train)
            forecasttrainY2 = fanNormalized(forecasttrainY2, res_Max_min[2])
            actualtrainY2 = fanNormalized(Y_train, res_Max_min[2])
            rmse_fan_train = sqrt(mean_squared_error(forecasttrainY2, actualtrainY2))
            print('steps nums: %d, Test RMSE_train: %.3f,RMSEfan_train: %.3f' % (i,rmse_train,rmse_fan_train))

            forecasttestY2 = model.predict([X_test,TM_test])
            rmse_test = mean_squared_error(forecasttestY2, Y_test)
            forecasttestY2 = fanNormalized(forecasttestY2, res_Max_min[2])
            actualtestY2 = fanNormalized(Y_test, res_Max_min[2])
            rmse_fan_test = sqrt(mean_squared_error(forecasttestY2, actualtestY2))
            print('steps nums: %d,Test rmse_test: %.3f,rmse_fan_test: %.3f' % (i,rmse_test,rmse_fan_test))
    # for_test_Y_2 = model.predict([res_test_2[0],res_test_2[1]])
    # for_test_Y_2 = fanNormalized(for_test_Y_2, res_Max_min[2])
    # act_test_Y_2 = fanNormalized(res_test_2[2], res_Max_min[2])
    # pyplot.plot(act_test_Y_2, label='actY_test')
    # pyplot.plot(for_test_Y_2, label='forY_test')
    # pyplot.legend()
    # pyplot.show()
    nums=nums+1
    print('The Training cycle model nums is: %d' % (nums))
    model.save('Rmodel/'+str(nums)+'my_model.h5')
    return model
 

def model_fine(timesteps, domin, optimizer_chose, epochs,nums_save):
    res = []
    input_y = Input(shape=(timesteps, domin))

    lstm = GRU(32, return_sequences=False)  # , kernel_initializer='Orthogonal', bias_initializer='zeros'
    lSTM_dense1 = Dense(128, activation='selu', name='LSTM_output3')
    lSTM_dense2 = Dense(64, activation='selu', name='LSTM_output4')
    lSTM_dense3 = Dense(1, activation='linear', name='LSTM_output5')

    lstm_1 = lstm(input_y)
    lstm_1 = lSTM_dense1(lstm_1)
    lstm_1 = lSTM_dense2(lstm_1)
    lstm_y_out = lSTM_dense3(lstm_1)


    lSTM_model = model(input_y, lstm_y_out)
    lSTM_model.summary()

    lSTM_model.compile(optimizer=optimizer_chose, loss = 'mse' )
    for i in range(epochs):
        time_start = time.time()
        history = lSTM_model.fit(X_train, Y_train, epochs=1,batch_size=128,verbose=0, shuffle=True) #validation_split=0.2,
        time_end = time.time()
        time_train_cost=time_end - time_start
        if i % 20 == 0 :
            # print("============================================================")
            time_predict_start=time.time()
            forecasttrainY2 = lSTM_model.predict(X_train)
            forecasttestY2 = lSTM_model.predict(X_test)
            forecasttestfh = lSTM_model.predict(X_test_fh)
            time_predict_end=time.time()
            time_predict_cost=time_predict_end-time_predict_start
            with open('result_GRU/TimeCost_'+str(nums_save)+'.txt', 'a') as f:
                f.write('step:'+str(i)+'   time_train_cost:'+str(round(time_train_cost, 4))+'   time_predict_cost:'+str(round(time_predict_cost, 4)) +'\n')

            rmse_train = mean_squared_error(forecasttrainY2, Y_train)
            forecasttrainY2 = fanNormalized(forecasttrainY2, res_Max_min[2])
            actualtrainY2 = fanNormalized(Y_train, res_Max_min[2])
            rmse_fan_train = sqrt(mean_squared_error(forecasttrainY2, actualtrainY2))
            mae_fan_train = mean_absolute_error(forecasttrainY2, actualtrainY2)
            R2_train= r2_score(forecasttrainY2, actualtrainY2)

            print('=============steps nums: %d, Test RMSE_train: %.5f,RMSEfan_train: %.5f' % (i,rmse_train,rmse_fan_train))

            rmse_test = mean_squared_error(forecasttestY2, Y_test)
            forecasttestY2 = fanNormalized(forecasttestY2, res_Max_min[2])
            actualtestY2 = fanNormalized(Y_test, res_Max_min[2])
            rmse_fan_test = sqrt(mean_squared_error(forecasttestY2, actualtestY2))

            mae_fan_test = mean_absolute_error(forecasttestY2, actualtestY2)
            R2_test= r2_score(forecasttestY2, actualtestY2)


            write_mat(forecasttestY2,'result_GRU/data/', str(nums_save)+'_'+str(i)+'forecasttestY2')
            write_mat(actualtestY2,'result_GRU/data/', 'actualtestY2_'+str(nums_save))

            # dataset = io.loadmat('result/data/'+str(nums_save)+'_'+str(i)+'forecasttestY2.mat')
            # forecasttestY21 = dataset['data']
            # dataset = io.loadmat('result/data/actualtestY2_'+str(nums_save)+'.mat')
            # actualtestY21 = dataset['data']

            print('===========steps nums: %d,Test rmse_test: %.5f,rmse_fan_test: %.5f' % (i,rmse_test,rmse_fan_test))

            rmse_test_fh = mean_squared_error(forecasttestfh, Y_test_fh)
            forecasttestfh = fanNormalized(forecasttestfh, res_Max_min[2])
            actualtestfh = fanNormalized(Y_test_fh, res_Max_min[2])
            rmse_fan_test_fh = sqrt(mean_squared_error(forecasttestfh, actualtestfh))
            mae_fan_test_fh = mean_absolute_error(forecasttestfh, actualtestfh)
            R2_test_fh= r2_score(forecasttestfh, actualtestfh)

            write_mat(forecasttestfh,'result_GRU/data/', str(nums_save)+'_'+str(i)+'forecasttestfh')
            write_mat(actualtestfh,'result_GRU/data/', 'actualtestfh'+str(nums_save))


            with open('result_GRU/result_'+str(nums_save)+'.txt', 'a') as f:
                f.write('step:'+str(i)+'    RMSE_train:'+str(round(rmse_fan_train, 4)) +'  mae_train:'+str(round(mae_fan_train, 4)) +'  R2_train:'+str(round(R2_train, 4))+'    RMSE_test:'+str(round(rmse_fan_test, 4)) +'  mae_test:'+str(round(mae_fan_test, 4)) +'  R2_test:'+str(round(R2_test, 4))+'    RMSE_test_fh:'+str(round(rmse_fan_test_fh, 4)) +'  mae_test_fh:'+str(round(mae_fan_test_fh, 4)) +'  R2_test_fh:'+str(round(R2_test_fh, 4))+'\n')

            pyplot.plot(actualtestY2, label='actualtestY')
            pyplot.plot(forecasttestY2, label='forecasttestY')
            pyplot.legend()
            pyplot.savefig("result_GRU/result_"+str(nums_save)+"/number_"+str(i)+".png")
            pyplot.close()

    # lSTM_model.save('model/'+str(nums_save)+'my_model.h5')
    # #画离散训练集
    # pyplot.plot(actualtrainY2, label='actualtestY_train')
    # pyplot.plot(forecasttrainY2, label='forecasttestY_train')
    # pyplot.legend()
    # pyplot.show()
    # # #画离散验证集
    # pyplot.plot(actualtestY2, label='actualtestY_train')
    # pyplot.plot(forecasttestY2, label='forecasttestY_train')
    # pyplot.legend()
    # pyplot.show()
    # #画连续测试集
    # forecasttest_Y_2 = lSTM_model.predict([X_test_fh,TM_test_fh[:,0]])
    # forecasttest_Y_2 = fanNormalized(forecasttest_Y_2, res_Max_min[2])
    # actualtest_Y_2 = fanNormalized(Y_test_fh[:,1].reshape(-1,1), res_Max_min[2])
    # pyplot.plot(actualtest_Y_2, label='actY_test')
    # pyplot.plot(forecasttest_Y_2, label='forY_test')
    # pyplot.legend()
    # pyplot.show()
    return lSTM_model

def selectdata_run(dataset,max_value,min_value,limitlength):
    datasetindexlest=[]
    start_index=0
    count=0
    flag=0
    for i in range(len(dataset)):
        if(dataset[i]>=min_value and dataset[i]<=max_value):
            if(flag==0):
                start_index=i
            count=count+1
            flag=1
        else:
            if(flag==1):
                flag = 0
                if(count>=limitlength):
                    datasetindexlest.append([start_index, count+start_index])
                    start_index = start_index + count
                    count = 0
                else:
                    count=0
    return  datasetindexlest

def get_data_time(ans,timesteps,domin,col_goal,values):
    res=[]
    result_up_x=[]
    result_up_y=[]
    result_up_y1 = []
    for n in range(len(ans)):
        data_length=abs(ans[n][0]-ans[n][1])
        if(data_length>=25):
            for m in range(data_length-timesteps):
                temp_x = []
                temp_y = []
                temp_y1 = []
                for i in range(timesteps):
                    temp_x.append(values[ans[n][0] + m + i, 0:domin])
                    temp_y.append(values[ans[n][0] + m + i, col_goal])
                    temp_y1.append(values[ans[n][0] + m + i, 8])
                temp_x_ = np.array(temp_x)
                temp_y_ = np.array(temp_y)
                temp_y1_ = np.array(temp_y1)
                result_up_x.append(temp_x_)
                result_up_y.append(temp_y_)
                result_up_y1.append(temp_y1_)
    result_x=np.array(result_up_x)
    s1_x=result_x.reshape(len(result_up_x)*timesteps,domin)
    s1_y=np.array(result_up_y)[:,timesteps-1].reshape(len(result_up_y),1)
    s1_y1=np.array(result_up_y1)[:,timesteps-1].reshape(len(result_up_y),1)
    res.append(s1_x)
    res.append(s1_y)
    res.append(s1_y1)
    return res

def train_test(X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver,xuanze):
    x_test,tm_test,y_test,y_jl_test,load_test = X_k_ver[xuanze],TM_k_ver[xuanze],Y_k_ver[xuanze],Y_jl_ver[xuanze],Load_k_ver[xuanze]
    x_train,tm_train,y_train,y_jl_train,load_train = [],[],[],[],[]
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
            else:
                x_train = np.vstack(([x_train,X_k_ver[i]]))
                tm_train = np.vstack(([tm_train, TM_k_ver[i]]))
                y_train = np.vstack(([y_train,Y_k_ver[i]]))
                y_jl_train=np.vstack(([y_jl_train,Y_jl_ver[i]]))
                load_train=np.vstack(([load_train,Load_k_ver[i]]))
    return x_train,tm_train,y_train,y_jl_train,load_train,x_test,tm_test,y_test,y_jl_test,load_test

def train_data_cycle(X_dataset,Tm_dataset,Y_dataset,Y_jl,Load_dataset):
    # index = [i for i in range(Y_dataset.shape[0])]
    fenshu = np.int(Y_dataset.shape[0] / number_fs)
    # np.random.shuffle(index)
    # X_select = X_dataset[index, :, :]
    # TM_select = Tm_dataset[index, :]
    # Y_select = Y_dataset[index, :]
    X_select = X_dataset
    TM_select = Tm_dataset
    Y_select = Y_dataset
    Y_jl_select = Y_jl
    Load_select=Load_dataset
    X_k_ver, TM_k_ver, Y_k_ver,Y_jl_ver,Load_k_ver = [], [], [], [], []
    for i in range(number_fs):
        if i < number_fs-1:
            X_k_ver.append(X_select[i * fenshu:fenshu * (i + 1), :, :])
            TM_k_ver.append(TM_select[i * fenshu:fenshu * (i + 1), :])
            Y_k_ver.append(Y_select[i * fenshu:fenshu * (i + 1), :])
            Y_jl_ver.append(Y_jl_select[i * fenshu:fenshu * (i + 1), :])
            Load_k_ver.append(Load_select[i * fenshu:fenshu * (i + 1), :])
        else:
            X_k_ver.append(X_select[i * fenshu:Y_dataset.shape[0], :, :])
            TM_k_ver.append(TM_select[i * fenshu:Y_dataset.shape[0],:])
            Y_k_ver.append(Y_select[i * fenshu:Y_dataset.shape[0],:])
            Y_jl_ver.append(Y_jl_select[i * fenshu:Y_dataset.shape[0],:])
            Load_k_ver.append(Load_select[i * fenshu:Y_dataset.shape[0],:])

    return X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver

if __name__ == '__main__':

    # 按照步长划分无状态的数据段
    timesteps = 40
    Dropout_rate = 0.50
    epochs = 300
    domin = 6  # 模型输入维度 输入
    col_goal=6 #模型输出（蒸汽出口温度）的索引
    Load_index = 8  # 负荷的索引

    """  从matlab生成的选择后的数据  """

    dataset=io.loadmat('data/from matlab/data_Z1.mat')

    data_all=dataset["data_out_Z_1"]
    s12=[]
    s12.append(data_all[:,0:6])
    s12.append(data_all[:, 6:8]) # 出口温度实测值和模型输出值
    s12.append(data_all[:, 8]) # 用于机理约束的负荷
    s12.append(data_all[:, 9:26]) # 用于机理约束的壁温测点及平均值
    # 划分测试级，训练集和泛化区验证
    length_Train_test=29000
    X_train_test,Y_train_test,Load_train_test,TM_train_test,X_test_fh,Y_test_fh1,Load_test_fh,TM_test_fh,res_Max_min = Model_input(s12, length_Train_test, timesteps, domin)  #归一化及上下限

    X_test_fh = X_test_fh
    TM_test_fh =TM_test_fh
    Y_test_fh = np.array(Y_test_fh1[:,0]).reshape(-1,1)
    Y_jl_fh = np.array(Y_test_fh1[:,1]).reshape(-1,1)
    Load_test_fh = Load_test_fh



    #循环数据集
    X_dataset = X_train_test
    Tm_dataset =TM_train_test
    Y_dataset = np.array(Y_train_test[:,0]).reshape(-1,1)
    Y_jl = np.array(Y_train_test[:,1]).reshape(-1,1)
    Load_dataset = Load_train_test

    number_fs=10 # 完整数据集分为多少份
    X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver = train_data_cycle(X_dataset,Tm_dataset,Y_dataset,Y_jl,Load_dataset)

    # test_weizhi=number_fs-1
    # X_train,TM_train,Y_train,Y_jl_train,Load_train,X_test, TM_test,Y_test,Y_jl_test,Load_test =train_test(X_k_ver,TM_k_ver,Y_k_ver,Y_jl_ver,Load_k_ver, test_weizhi)


    # result_ = model_fine(timesteps, domin, RMSprop, epochs, test_weizhi)

    sgd = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    RMSprop = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=1e-06)
    ADam = optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08)

    Updata_model=0
    for i in range(4, 5):
        print("i====//////////////////////////============" + str(i))
        # i 是选择十段数据（打乱）中的一段，以此作为测试集
        X_train,TM_train,Y_train,Y_jl_train,Load_train,X_test, TM_test,Y_test,Y_jl_test,Load_test= train_test(
            X_k_ver, TM_k_ver, Y_k_ver, Y_jl_ver, Load_k_ver, i)
        if i==0:
            print('训练集：%.3f', Y_train.shape[0])
            print('测试集：%.3f', Y_test.shape[0])
            print('泛化验证集：%.3f', Y_test_fh.shape[0])

        if Updata_model==0:
            # 训练数据集，每次模型更新
            result = model_fine(timesteps, domin, RMSprop, epochs, i)
        else:
            # 训练数据集，每次模型不更新
            result_ = Model_cycle(result_, epochs, i)