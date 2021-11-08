function [re,pr] = get_Re_Pr(pp,u,d,ur,cp,lamda)     
% pp 密度
% u 速度
% d 内直径
% ur 运动粘度
% cp 定压比热容
% lamda 水的导热系数
   re=u*d/ur;
%转换为动力粘度
   ur=ur*pp;
   pr=ur*cp/lamda; 
  
end