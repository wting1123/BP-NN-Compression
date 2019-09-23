
clc % ����
clear % �����ڴ�
Image=imread('d:\matlab\test.png'); % ��ȡͼ��
%Image=rgb2groy(Image);��ͼ���Ϊ�Ҷ�ͼ��
Image=imresize(Image,[128,128]); % ��ͼ��ת��Ϊ128*128���ش�С
figure(1) %��������һ
imshow(Image),title('ԭͼ��');
P=[]; %�������

for i=1:32
    for j=1:32
        Element_Image=Image((i-1)*4+1:i*4,(j-1)*4+1:j*4);
        Vector=double(reshape(Element_Image,16,1)); %��������
        P_Vector=Vector/255; %��һ��������֤ϵͳ���ȶ���
        P=[P,P_Vector]; %16*1024
    end
end

Input=P'; %Pת��Ϊ1024*16
Output=Input;  %�������=����

tic   %��ʱ

%ѹ���� 16/8=2
HiddenLayer_Weight=double(rand(16,8)*2-1); %��ʼ��������HiddenLayer_WeightȨֵ����Χ-1~1
OutputLayer_Weight=double(rand(8,16)*2-1); %��ʼ�������OutputLayer_WeightȨֵ����Χ-1~1

HiddenLayerWeight_Previous=double(zeros(16,8));
OutputLayerWeight_Previous=double(zeros(8,16));

lr=0.01; %ѧϰ��Ϊ0-1��С��
a=0.35; %��������
error_temp=[]; %�������ʱ����ŵ�һ�ζ��������ͽ������
error=ones(10000,1); %����ѵ�������仯�������󣬳�ʼֵȫΪ1

for epochs=1:10000
    %�����������������������Sigmoid������f(x)=1/[1+e^(-x)]
    HiddenLayer=1./(1+exp(-(Input*HiddenLayer_Weight))); %���������HiddenLayerΪ1024*8����
    OutputLayer=1./(1+exp(-(HiddenLayer*OutputLayer_Weight))); %��������OutputLayerΪ1024*16����

    OutputLayer_delta=(Output-OutputLayer).*OutputLayer.*(1-OutputLayer); %1024*16
    HiddenLayer_delta=OutputLayer_delta*OutputLayer_Weight'.*HiddenLayer.*(1-HiddenLayer); %1024*8

    OutputLayerWeight_Change=(1-a)*lr*HiddenLayer'*(OutputLayer_delta)+a*OutputLayerWeight_Previous; %8*16
    HiddenLayerWeight_Change=(1-a)*lr*Input'*(HiddenLayer_delta)+a*HiddenLayerWeight_Previous; %16*8

    OutputLayer_Weight=OutputLayer_Weight+OutputLayerWeight_Change;
    HiddenLayer_Weight=HiddenLayer_Weight+HiddenLayerWeight_Change;
    
    OutputLayerWeight_Previous=OutputLayerWeight_Change;
    HiddenLayerWeight_Previous=HiddenLayerWeight_Change;
    
    
    error_temp=sum((Output-OutputLayer).*(Output-OutputLayer))'; %1024*16��sum������ͣ���ͺ���Ϊ1*16������ת��Ϊ16*1
    error(epochs)=sqrt(sum(error_temp)/(16*1024)); %�����׼��
      
    if error(epochs)<=0.005
        break
    end
end

toc

figure(3)
plot(error);

%���棬�Ѿ�ѵ���õ����磬Input������
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

YRefactor_t=uint8(YYRefactor_test*255); %����һ��
figure(2)
imshow(YRefactor_t),title('�ؽ�ͼ��');





