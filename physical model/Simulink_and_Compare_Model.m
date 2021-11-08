
%% 仿真
TP=length(x);%仿真时间
opt=simset('solver','ode14x');
% 
% [tout0,xout0,y]=sim('Zmixed',[1 TP],opt);%没有传热系数计算，通过经验系数代替
% 
% [tout0,xout0,y_new1]=sim('Zmixed_new1',[1 TP],opt);%旧的传热系数计算公式
% 
% [tout0,xout0,y_new2]=sim('Zmixed_new2',[1 TP],opt);%新的传热系数计算公式
if First_Simulink==0
    sim('Zmixed',[1 TP],opt);%没有传热系数计算，通过经验系数代替
    sim('Zmixed_new1',[1 TP],opt);%旧的传热系数计算公式
    sim('Zmixed_new2',[1 TP],opt);%新的传热系数计算公式
    First_Simulink=1;
else
    
end

%%画图
figure;
% plot(x);%负荷
% hold on;
plot(simout_Tf_db2.signals.values(300:end),'DisplayName','duibi2');% 旧的传热系数计算公式
hold on;
plot(simout_Tf_new.signals.values(300:end),'DisplayName','lixiang');% 新的传热系数计算公式
hold on;
plot(T1(300:end),'DisplayName','实测值');%实际温度
hold off;

%%画图
figure;
% plot(x);%负荷
% hold on;
plot(simout_Tf_db1.signals.values(300:end),'DisplayName','duibi1');% 没有传热系数计算，通过经验系数代替
hold on;
plot(simout_Tf_new.signals.values(300:end),'DisplayName','lixiang');% 新的传热系数计算公式
hold on;
plot(T1(300:end),'DisplayName','实测值');%实际温度
hold off;

aa=1;





%% 画图 
% scope_start=1;
% scope_end=1000;
% [tout0,xout0,y]=sim('compare_result',[1 TP],opt);%输出多个y值
