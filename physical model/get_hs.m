function hs=get_hs(pp,u,d,ur,cp,lamda)
%pp密度
%u速度


[re,pr] = get_Re_Pr(pp,u,d,ur,cp,lamda);
hs=0.013*lamda/d*(re)^0.867*pr^(1/3);
hs=hs*1000;
%对于其他定义的注解情况
% h_s=0.018 *lamda/d*(re)^(-0.25)*(re-500)^1.07 * pr^0.42*(pr/1 )^0.11;
end