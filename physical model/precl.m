function y = precl(u2,u1)
u1=u1+273.15;
u2=u2*1000;
D=refpropm('H','T',u1,'P',u2,'water');
y = D/1000;% ‰≥ˆÏ KJ/Kg
if y<2000
    s=3;
end