function y = preclP(u2,u1)
u1=u1+273.15;
u2=u2*1000;
D=refpropm('P','T',u1,'H',u2,'water');
y = D/1000;% ‰≥ˆÏ KJ/Kg