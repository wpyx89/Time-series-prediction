function y = precllamda(T,P)
T=T+273.15;
P=P*1000;
L=refpropm('L','T',T,'P',P,'water');
y = L/1000;%�������ϵ��W/m/K��ע�⣩