function hs=get_hs(pp,u,d,ur,cp,lamda)
%pp�ܶ�
%u�ٶ�


[re,pr] = get_Re_Pr(pp,u,d,ur,cp,lamda);
hs=0.013*lamda/d*(re)^0.867*pr^(1/3);
hs=hs*1000;
%�������������ע�����
% h_s=0.018 *lamda/d*(re)^(-0.25)*(re-500)^1.07 * pr^0.42*(pr/1 )^0.11;
end