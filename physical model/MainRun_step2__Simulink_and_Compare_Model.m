
%% ����
TP=length(x);%����ʱ��
opt=simset('solver','ode14x');
% 
% [tout0,xout0,y]=sim('Zmixed',[1 TP],opt);%û�д���ϵ�����㣬ͨ������ϵ������
% 
% [tout0,xout0,y_new1]=sim('Zmixed_new1',[1 TP],opt);%�ɵĴ���ϵ�����㹫ʽ
% 
% [tout0,xout0,y_new2]=sim('Zmixed_new2',[1 TP],opt);%�µĴ���ϵ�����㹫ʽ
if First_Simulink==0
    sim('Zmixed',[1 TP],opt);%û�д���ϵ�����㣬ͨ������ϵ������
    sim('Zmixed_new1',[1 TP],opt);%�ɵĴ���ϵ�����㹫ʽ
    sim('Zmixed_new2',[1 TP],opt);%�µĴ���ϵ�����㹫ʽ
    First_Simulink=1;
else
    
end

%%��ͼ
figure;
% plot(x);%����
% hold on;
plot(simout_Tf_db2.signals.values(300:end),'DisplayName','duibi2');% �ɵĴ���ϵ�����㹫ʽ
hold on;
plot(simout_Tf_new.signals.values(300:end),'DisplayName','lixiang');% �µĴ���ϵ�����㹫ʽ
hold on;
plot(T1(300:end),'DisplayName','ʵ��ֵ');%ʵ���¶�
hold off;

%%��ͼ
figure;
% plot(x);%����
% hold on;
plot(simout_Tf_db1.signals.values(300:end),'DisplayName','duibi1');% û�д���ϵ�����㣬ͨ������ϵ������
hold on;
plot(simout_Tf_new.signals.values(300:end),'DisplayName','lixiang');% �µĴ���ϵ�����㹫ʽ
hold on;
plot(T1(300:end),'DisplayName','ʵ��ֵ');%ʵ���¶�
hold off;

aa=1;





%% ��ͼ 
% scope_start=1;
% scope_end=1000;
% [tout0,xout0,y]=sim('compare_result',[1 TP],opt);%������yֵ
