clear;
clc;

path_read='用于机理模型的数据/';
path_write='用于python混合模型的数据/';

%% 加载所用数据

% % 一种磨煤机运行模式的经过初步筛选的数据

load([path_read,'data_physical.mat']);
data=data_physical;

First_Simulink=0;

%% 各变量的物理意义
IT=length(data(:,1));
Lp=data(1:IT,1);%左侧出口压力
Rp=data(1:IT,2);%右侧出口压力
Lt=data(1:IT,3);%左侧出口温度
Rt=data(1:IT,4);%右侧出口温度
Zp=data(1:IT,5);%总压
Zt=data(1:IT,6);%总温度
% Zf=data(1:IT,7);%总流量
% Lp1=data(1:IT,8);%末级过热器入口压力
% Lp2=data(1:IT,9);%末级过热器出口压力
% InputWP=data(1:IT,10);%减温水压力
% InputWT=data(1:IT,11);%减温水温度
Inputp2=data(1:IT,12);%末级过热器入口压力
InputT1=data(1:IT,13);%末级过热器入口温度
T1=data(1:IT,14);%屏式过热器出口温度
Wtf=data(1:IT,15);%二级减温水流量
Tf1=data(1:IT,16);%入口蒸汽温度
% Tg1=data(1:IT,17);%锅炉出口烟温
% Tamb=data(1:IT,18);%环境温度
% WG=data(1:IT,19);%风量
% WR=data(1:IT,20);%煤量
M_wall=data(1:IT,21);%壁温约束
% x=data(1:IT,22);%负荷
Tm=mean(data(1:IT,23:38),2);%实际壁温取16个壁温的平均值
T_all=data(1:IT,23:38);
% B1=data(1:IT,23);%壁温1
% B2=data(1:IT,24);%壁温2
% B3=data(1:IT,25);%壁温3
% B4=data(1:IT,26);%壁温4
% B5=data(1:IT,27);%壁温5
% B6=data(1:IT,28);%壁温6
% B7=data(1:IT,29);%壁温7
% B8=data(1:IT,30);%壁温8
% B9=data(1:IT,31);%壁温9
% B10=data(1:IT,32);%壁温10
% B11=data(1:IT,33);%壁温11
% B12=data(1:IT,34);%壁温12
% B13=data(1:IT,35);%壁温13
% B14=data(1:IT,36);%壁温14
% B15=data(1:IT,37);%壁温15
% B16=data(1:IT,38);%壁温16
%% 磨煤机筛选
milla=data(1:IT,39);%磨煤机A
millb=data(1:IT,40);%磨煤机B
millc=data(1:IT,41);%磨煤机C
milld=data(1:IT,42);%磨煤机D
mille=data(1:IT,43);%磨煤机E
millf=data(1:IT,44);%磨煤机F

%% 屏过simulink输入数据 运行Zmixed.xls程序
Tin_R=data(1:IT,45);%高过入口蒸汽温度_R
Tin_L=data(1:IT,46);%高过入口蒸汽温度_L
Tout_L=data(1:IT,47);%主蒸汽温度
Tout_R=data(1:IT,48);%主蒸汽温度
x=data(1:IT,49);%负荷
Zf=data(1:IT,50);%总流量
WG=data(1:IT,51);%风量
WR=data(1:IT,52);%煤量
Tg1=data(1:IT,53);%锅炉出口烟温
Tg_d_in=data(1:IT,54);%低过入口烟温
Tamb=data(1:IT,55);%环境温度
Lp1=data(1:IT,56);%末级过热器入口压力
Lp2=data(1:IT,57);%末级过热器出口压力
T1_sp_r=data(1:IT,58);%右侧一减温度设定值
T1_sp_l=data(1:IT,59);%一级过热器温度设定值
T2_sp_r=data(1:IT,60);%右侧二减温度设定值
T2_sp_l=data(1:IT,61);%二级过热器温度设定值
GT1_l=data(1:IT,62);%阀门左1
GT2_l=data(1:IT,63);%阀门左2
GT1_r=data(1:IT,64);%阀门右1
GT2_r=data(1:IT,65);%阀门右2
InputWP=data(1:IT,66);%减温水压力
InputWT=data(1:IT,67);%减温水温度
T1_in_l=data(1:IT,68);%左侧一级过热器入口温度
T1_out_l=data(1:IT,69);%左侧一级过热器出口温度
T2_in_l=data(1:IT,70);%左侧二级过热器入口温度
T2_out_l=data(1:IT,71);%左侧二级过热器出口温度
T1_in_r=data(1:IT,72);%右侧一级过热器入口温度
T1_out_r=data(1:IT,73);%右侧一级过热器出口温度
T2_in_r=data(1:IT,74);%右侧二级过热器入口温度
T2_out_r=data(1:IT,75);%右侧二级过热器出口温度
WaterP_sub=data(1:IT,76);%给水差压
Water_l1=data(1:IT,77);%左侧一级流量
Water_l2=data(1:IT,78);%左侧二级流量
Water_r1=data(1:IT,79);%右侧一级流量
Water_r2=data(1:IT,80);%右侧二级流量
Tm_G=mean(data(1:IT,81:102),2);%壁温
% Tg_m_1=data(1:IT,81);%壁温g1
% Tg_m_2=data(1:IT,82);%壁温g2
% Tg_m_3=data(1:IT,83);%壁温g3
% Tg_m_4=data(1:IT,84);%壁温g4
% Tg_m_5=data(1:IT,85);%壁温g5
% Tg_m_6=data(1:IT,86);%壁温g6
% Tg_m_7=data(1:IT,87);%壁温g7
% Tg_m_8=data(1:IT,88);%壁温g8
% Tg_m_9=data(1:IT,89);%壁温g9
% Tg_m_10=data(1:IT,90);%壁温g10
% Tg_m_11=data(1:IT,91);%壁温g11
% Tg_m_12=data(1:IT,92);%壁温g12
% Tg_m_13=data(1:IT,93);%壁温g13
% Tg_m_14=data(1:IT,94);%壁温g14
% Tg_m_15=data(1:IT,95);%壁温g15
% Tg_m_16=data(1:IT,96);%壁温g16
% Tg_m_17=data(1:IT,97);%壁温g17
% Tg_m_18=data(1:IT,98);%壁温g18
% Tg_m_19=data(1:IT,99);%壁温g19
% Tg_m_20=data(1:IT,100);%壁温g20
% Tg_m_21=data(1:IT,101);%壁温g21
% Tg_m_22=data(1:IT,102);%壁温g22