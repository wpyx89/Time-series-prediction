%% python�������� ��ǰ��
%���룺����	��������������¶�	���� ����ˮѹ�� 
%���������������¶�
data_out_D=[x,T1_in_l,GT1_l,InputWP,T1_out_l];
save ([path_write,'data_D.mat'], 'data_out_D');
%% python�������� �ͺ���  &�޻���
%���룺���� ú�� ����ѹ	����	��������������¶�(T1_out_l/Tf1)
%����������������¶�
data_out_Z_0=[x,WR,Zp,WG,Tf1,T1];
save ([path_write,'data_Z_0.mat'], 'data_out_Z_0');




%% python�������� �ͺ���  &���� ���ϵ�� �޻��ȹ�ʽ
%���룺�磬ý�����£����������������(ģ�����)������¶ȣ�ѹ��(p2)�������������¶�(ʵ��)�������������¶�(����ģ��)�����ɣ�����ƽ��ֵ��16������
%����������������¶�
data_out_Z_1=[WG,WR,Tg1,simout3.signals.values(1:end),Tf1,Inputp2,T1,simout_Tm_db1.signals.values(1:end),x,Tm,T_all];
save ([path_write,'data_Z1.mat'], 'data_out_Z_1');


%% python�������� �ͺ���  &���� ���»��ȹ�ʽ
%���룺�磬ý�����£����������������(ģ�����)������¶ȣ�ѹ��(p2)�������������¶�(ʵ��)�������������¶�(����ģ��)�����ɣ�����ƽ��ֵ��16������
%����������������¶�
data_out_Z_2=[WG,WR,Tg1,simout3.signals.values(1:end),Tf1,Inputp2,T1,simout_Tm_new.signals.values(1:end),x,Tm,T_all];
save ([path_write,'data_Z2.mat'], 'data_out_Z_2');


%% ����ģ�ͳ����¶ȶԱ�ֵ

data_duibi=[x,simout_Tf_db1.signals.values(1:end),simout_Tf_new.signals.values(1:end),T1];
save ([path_write,'data_Tf_all.mat'], 'data_duibi');