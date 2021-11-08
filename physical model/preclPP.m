function y = preclPP(u1,u2)
%u1ÎÂ¶È
% u2 Ñ¹Á¦
         u1=u1+273.15;
         u2=u2*1000;
         D=refpropm('D','T',u1,'P',u2,'water');
       y=D;