function [re,pr] = get_Re_Pr(pp,u,d,ur,cp,lamda)     
% pp �ܶ�
% u �ٶ�
% d ��ֱ��
% ur �˶�ճ��
% cp ��ѹ������
% lamda ˮ�ĵ���ϵ��
   re=u*d/ur;
%ת��Ϊ����ճ��
   ur=ur*pp;
   pr=ur*cp/lamda; 
  
end