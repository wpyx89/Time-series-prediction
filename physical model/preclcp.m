function y = preclcp(T,P)
T=T+273.15;
P=P*1000;
C=refpropm('C','T',T,'P',P,'water');
y = C;%�����ѹ������J/(Kg.K)��ע�⣩