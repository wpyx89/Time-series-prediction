clear;
clc;

path_read='���ڻ���ģ�͵�����/';
path_write='����python���ģ�͵�����/';

%% ������������

% % һ��ĥú������ģʽ�ľ�������ɸѡ������

load([path_read,'data_physical.mat']);
data=data_physical;

First_Simulink=0;

%% ����������������
IT=length(data(:,1));
Lp=data(1:IT,1);%������ѹ��
Rp=data(1:IT,2);%�Ҳ����ѹ��
Lt=data(1:IT,3);%�������¶�
Rt=data(1:IT,4);%�Ҳ�����¶�
Zp=data(1:IT,5);%��ѹ
Zt=data(1:IT,6);%���¶�
% Zf=data(1:IT,7);%������
% Lp1=data(1:IT,8);%ĩ�����������ѹ��
% Lp2=data(1:IT,9);%ĩ������������ѹ��
% InputWP=data(1:IT,10);%����ˮѹ��
% InputWT=data(1:IT,11);%����ˮ�¶�
Inputp2=data(1:IT,12);%ĩ�����������ѹ��
InputT1=data(1:IT,13);%ĩ������������¶�
T1=data(1:IT,14);%��ʽ�����������¶�
Wtf=data(1:IT,15);%��������ˮ����
Tf1=data(1:IT,16);%��������¶�
% Tg1=data(1:IT,17);%��¯��������
% Tamb=data(1:IT,18);%�����¶�
% WG=data(1:IT,19);%����
% WR=data(1:IT,20);%ú��
M_wall=data(1:IT,21);%����Լ��
% x=data(1:IT,22);%����
Tm=mean(data(1:IT,23:38),2);%ʵ�ʱ���ȡ16�����µ�ƽ��ֵ
T_all=data(1:IT,23:38);
% B1=data(1:IT,23);%����1
% B2=data(1:IT,24);%����2
% B3=data(1:IT,25);%����3
% B4=data(1:IT,26);%����4
% B5=data(1:IT,27);%����5
% B6=data(1:IT,28);%����6
% B7=data(1:IT,29);%����7
% B8=data(1:IT,30);%����8
% B9=data(1:IT,31);%����9
% B10=data(1:IT,32);%����10
% B11=data(1:IT,33);%����11
% B12=data(1:IT,34);%����12
% B13=data(1:IT,35);%����13
% B14=data(1:IT,36);%����14
% B15=data(1:IT,37);%����15
% B16=data(1:IT,38);%����16
%% ĥú��ɸѡ
milla=data(1:IT,39);%ĥú��A
millb=data(1:IT,40);%ĥú��B
millc=data(1:IT,41);%ĥú��C
milld=data(1:IT,42);%ĥú��D
mille=data(1:IT,43);%ĥú��E
millf=data(1:IT,44);%ĥú��F

%% ����simulink�������� ����Zmixed.xls����
Tin_R=data(1:IT,45);%�߹���������¶�_R
Tin_L=data(1:IT,46);%�߹���������¶�_L
Tout_L=data(1:IT,47);%�������¶�
Tout_R=data(1:IT,48);%�������¶�
x=data(1:IT,49);%����
Zf=data(1:IT,50);%������
WG=data(1:IT,51);%����
WR=data(1:IT,52);%ú��
Tg1=data(1:IT,53);%��¯��������
Tg_d_in=data(1:IT,54);%�͹��������
Tamb=data(1:IT,55);%�����¶�
Lp1=data(1:IT,56);%ĩ�����������ѹ��
Lp2=data(1:IT,57);%ĩ������������ѹ��
T1_sp_r=data(1:IT,58);%�Ҳ�һ���¶��趨ֵ
T1_sp_l=data(1:IT,59);%һ���������¶��趨ֵ
T2_sp_r=data(1:IT,60);%�Ҳ�����¶��趨ֵ
T2_sp_l=data(1:IT,61);%�����������¶��趨ֵ
GT1_l=data(1:IT,62);%������1
GT2_l=data(1:IT,63);%������2
GT1_r=data(1:IT,64);%������1
GT2_r=data(1:IT,65);%������2
InputWP=data(1:IT,66);%����ˮѹ��
InputWT=data(1:IT,67);%����ˮ�¶�
T1_in_l=data(1:IT,68);%���һ������������¶�
T1_out_l=data(1:IT,69);%���һ�������������¶�
T2_in_l=data(1:IT,70);%����������������¶�
T2_out_l=data(1:IT,71);%�����������������¶�
T1_in_r=data(1:IT,72);%�Ҳ�һ������������¶�
T1_out_r=data(1:IT,73);%�Ҳ�һ�������������¶�
T2_in_r=data(1:IT,74);%�Ҳ��������������¶�
T2_out_r=data(1:IT,75);%�Ҳ���������������¶�
WaterP_sub=data(1:IT,76);%��ˮ��ѹ
Water_l1=data(1:IT,77);%���һ������
Water_l2=data(1:IT,78);%����������
Water_r1=data(1:IT,79);%�Ҳ�һ������
Water_r2=data(1:IT,80);%�Ҳ��������
Tm_G=mean(data(1:IT,81:102),2);%����
% Tg_m_1=data(1:IT,81);%����g1
% Tg_m_2=data(1:IT,82);%����g2
% Tg_m_3=data(1:IT,83);%����g3
% Tg_m_4=data(1:IT,84);%����g4
% Tg_m_5=data(1:IT,85);%����g5
% Tg_m_6=data(1:IT,86);%����g6
% Tg_m_7=data(1:IT,87);%����g7
% Tg_m_8=data(1:IT,88);%����g8
% Tg_m_9=data(1:IT,89);%����g9
% Tg_m_10=data(1:IT,90);%����g10
% Tg_m_11=data(1:IT,91);%����g11
% Tg_m_12=data(1:IT,92);%����g12
% Tg_m_13=data(1:IT,93);%����g13
% Tg_m_14=data(1:IT,94);%����g14
% Tg_m_15=data(1:IT,95);%����g15
% Tg_m_16=data(1:IT,96);%����g16
% Tg_m_17=data(1:IT,97);%����g17
% Tg_m_18=data(1:IT,98);%����g18
% Tg_m_19=data(1:IT,99);%����g19
% Tg_m_20=data(1:IT,100);%����g20
% Tg_m_21=data(1:IT,101);%����g21
% Tg_m_22=data(1:IT,102);%����g22