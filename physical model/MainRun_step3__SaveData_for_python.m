%% python输入数据 导前区
%输入：负荷	减温器入口蒸汽温度	阀门 减温水压力 
%输出：过热器入口温度
data_out_D=[x,T1_in_l,GT1_l,InputWP,T1_out_l];
save ([path_write,'data_D.mat'], 'data_out_D');
%% python输入数据 滞后区  &无机理
%输入：负荷 煤量 主气压	风量	过热器入口蒸汽温度(T1_out_l/Tf1)
%输出：过热器出口温度
data_out_Z_0=[x,WR,Zp,WG,Tf1,T1];
save ([path_write,'data_Z_0.mat'], 'data_out_Z_0');




%% python输入数据 滞后区  &机理 拟合系数 无换热公式
%输入：风，媒，烟温，屏过入口蒸汽流量(模型输出)，入口温度，压力(p2)，过热器出口温度(实际)，过热器出口温度(机理模型)，负荷，壁温平均值，16个壁温
%输出：过热器出口温度
data_out_Z_1=[WG,WR,Tg1,simout3.signals.values(1:end),Tf1,Inputp2,T1,simout_Tm_db1.signals.values(1:end),x,Tm,T_all];
save ([path_write,'data_Z1.mat'], 'data_out_Z_1');


%% python输入数据 滞后区  &机理 最新换热公式
%输入：风，媒，烟温，屏过入口蒸汽流量(模型输出)，入口温度，压力(p2)，过热器出口温度(实际)，过热器出口温度(机理模型)，负荷，壁温平均值，16个壁温
%输出：过热器出口温度
data_out_Z_2=[WG,WR,Tg1,simout3.signals.values(1:end),Tf1,Inputp2,T1,simout_Tm_new.signals.values(1:end),x,Tm,T_all];
save ([path_write,'data_Z2.mat'], 'data_out_Z_2');


%% 机理模型出口温度对比值

data_duibi=[x,simout_Tf_db1.signals.values(1:end),simout_Tf_new.signals.values(1:end),T1];
save ([path_write,'data_Tf_all.mat'], 'data_duibi');