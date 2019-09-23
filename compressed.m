
clc % 清屏
clear % 清理内存
Image=imread('d:\matlab\test.png'); % 读取图像
%Image=rgb2groy(Image);将图像变为灰度图像
Image=imresize(Image,[128,128]); % 将图像转变为128*128像素大小
figure(1) %创建窗口一
imshow(Image),title('原图像');
P=[]; %输入矩阵

for i=1:32
    for j=1:32
        Element_Image=Image((i-1)*4+1:i*4,(j-1)*4+1:j*4);
        Vector=double(reshape(Element_Image,16,1)); %输入向量
        P_Vector=Vector/255; %归一化处理，保证系统的稳定性
        P=[P,P_Vector]; %16*1024
    end
end

Input=P'; %P转化为1024*16
Output=Input;  %理想输出=输入

tic   %计时

%压缩比 16/8=2
HiddenLayer_Weight=double(rand(16,8)*2-1); %初始化隐含层HiddenLayer_Weight权值，范围-1~1
OutputLayer_Weight=double(rand(8,16)*2-1); %初始化输出层OutputLayer_Weight权值，范围-1~1

HiddenLayerWeight_Previous=double(zeros(16,8));
OutputLayerWeight_Previous=double(zeros(8,16));

lr=0.01; %学习率为0-1的小数
a=0.35; %动量因子
error_temp=[]; %计算误差时，存放第一次对列误差求和结果矩阵
error=ones(10000,1); %随着训练次数变化的误差矩阵，初始值全为1

for epochs=1:10000
    %隐含层输出和输出层输出采用Sigmoid函数，f(x)=1/[1+e^(-x)]
    HiddenLayer=1./(1+exp(-(Input*HiddenLayer_Weight))); %隐含层输出HiddenLayer为1024*8矩阵
    OutputLayer=1./(1+exp(-(HiddenLayer*OutputLayer_Weight))); %输出层输出OutputLayer为1024*16矩阵

    OutputLayer_delta=(Output-OutputLayer).*OutputLayer.*(1-OutputLayer); %1024*16
    HiddenLayer_delta=OutputLayer_delta*OutputLayer_Weight'.*HiddenLayer.*(1-HiddenLayer); %1024*8

    OutputLayerWeight_Change=(1-a)*lr*HiddenLayer'*(OutputLayer_delta)+a*OutputLayerWeight_Previous; %8*16
    HiddenLayerWeight_Change=(1-a)*lr*Input'*(HiddenLayer_delta)+a*HiddenLayerWeight_Previous; %16*8

    OutputLayer_Weight=OutputLayer_Weight+OutputLayerWeight_Change;
    HiddenLayer_Weight=HiddenLayer_Weight+HiddenLayerWeight_Change;
    
    OutputLayerWeight_Previous=OutputLayerWeight_Change;
    HiddenLayerWeight_Previous=HiddenLayerWeight_Change;
    
    
    error_temp=sum((Output-OutputLayer).*(Output-OutputLayer))'; %1024*16用sum（列求和）求和后结果为1*16矩阵，再转置为16*1
    error(epochs)=sqrt(sum(error_temp)/(16*1024)); %计算标准差
      
    if error(epochs)<=0.005
        break
    end
end

toc

figure(3)
plot(error);

%仿真，已经训练好的网络，Input是输入
HiddenLayer=1./(1+exp(-(Input*HiddenLayer_Weight)));
OutputLayer=1./(1+exp(-(HiddenLayer*OutputLayer_Weight))); %1024*16
Y_Refactor=OutputLayer'; %16*1024

YRefactor_test=[];
for k=1:1024
    YRefactor_test1=reshape(Y_Refactor(:,k),4,4);
    YRefactor_test=[YRefactor_test,YRefactor_test1];
end

YYRefactor_test=[];
for k=1:32
    YYRefactor_test1=YRefactor_test(:,(k-1)*128+1:k*128);
    YYRefactor_test=[YYRefactor_test;YYRefactor_test1];
end

YRefactor_t=uint8(YYRefactor_test*255); %反归一化
figure(2)
imshow(YRefactor_t),title('重建图像');





