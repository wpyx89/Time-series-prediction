function y = precllamda(T,P)
T=T+273.15;
P=P*1000;
L=refpropm('L','T',T,'P',P,'water');
y = L/1000;%输出导热系数W/m/K（注意）