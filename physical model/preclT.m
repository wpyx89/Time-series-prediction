function y = preclT(u1,u2,u3)
if(u3<100)
    y=0;
else
    if (u1>4000)
    u1=4000;
    end
    if(u1<=100)
        y=-1;
    else
         u1=u1*1000;
         u2=u2*1000;
         D=refpropm('T','H',u1,'P',u2,'water');
         y = D-273.15;%ÎÂ¶È
    end
end


