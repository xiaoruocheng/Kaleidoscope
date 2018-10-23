clc
%--------------------------------����ת��---------------------------------
%---------------------��train.csv��test.csv������������һ��
str=['male'];
for i=1:1309
    Sex0(i,1)=strcmp(Sex(i,:),str)    % �Ա� ��-0 Ů-1 ת��
    if(Sex0(i,:))
    Sex1(i,1)=0;
    else
    Sex1(i,1)=1;
    end
    
str2=char(Embarked(i,1));   %�ۿ�ת�� C Q S- 1 2 3
switch(str2)
    case'C'
        Embarked1(i,1)=1;
    case'Q' 
        Embarked1(i,1)=2;
	case'S'
        Embarked1(i,1)=3;
end

count(i,1)=SibSp(i,1)+Parch(i,1);  %�Ƿ�������ͬ�� ��1 ��0
if count(i,1)>0
    count(i,1)=1;
end
end
% ����������-1���
Age0=Age;
Age0(isnan(Age0(:,1)))=[-1];
%---------------------------------------����-----------------------------------------
A=[ Pclass(1:891,1), Sex1(1:891,1), count(1:891,1), Embarked1(1:891,1), Age0(1:891,1)];
% ����ͬ��ֿ� A1 A2�ֱ�Ϊ�����ʹ��Ⱥ�� A3Ϊ����Ⱥ�壬M1 M2�ֱ�Ϊ��Ⱥ����������ĸ��� 
%���������̬�ֲ� ��mu1 mu2 sigma1 sigma2�ֱ�Ϊ����ľ�ֵ�ͷ���
j=1,k=1,m=1,n=1;
for i=1:891
    if(Survived(i,1)==0)
        A1(k,1:5)=A(i,1:5);
        if(A1(k,5)>-1)
            M1(m,1)=A1(k,5);
            m=m+1;
        end
        k=k+1;
    else
        A2(j,1:5)=A(i,1:5);
          if(A2(j,5)>-1)
            M2(n,1)=A2(j,5);
            n=n+1;
          end
          j=j+1;
    end
end
A3=[ Pclass(892:1309,1), Sex1(892:1309,1), count(892:1309,1), Embarked1(892:1309,1), Age0(892:1309,1)];
[mu1,sigma1]=normfit(M1);
[mu2,sigma2]=normfit(M2);
%-------------------------���������ɢ����ֵ-----------------------------------------       
P=length(A1)/length(A); %����������� P_aliveΪ������
P_alive=1-P;

P1=zeros(3,2,4); %��ѵ�����ݸ��������� Pclass Sex1 count Emparked1 ��P1 P2�ֱ��ʾ�����ʹ�������
P2=zeros(3,2,4);
for i=1:4
    B=tabulate(A1(:,i));
    C=B(:,[1 3]);
    d=max(size(C));
    if(d==3)
        P1(:,:,i)=C(:,:);
    else
        P1(1:2,:,i)=C(:,:);
        P1(3,1,i)=2;
    end
     P1(:,2,i)= P1(:,2,i)/100;
end
for i=1:4
    B=tabulate(A2(:,i));
    C=B(:,[1 3]);
    d=max(size(C));
    if(d==3)
        P2(:,:,i)=C(:,:);
    else
        P2(1:2,:,i)=C(:,:);
        P2(3,1,i)=2;
    end
     P2(:,2,i)= P2(:,2,i)/100;
end
      
 
% ���������� postΪ������� prodtΪ��Ȼ���� ֤��������ͬ ����������˶�������������Ȼ�����˻�
% ����equal������ÿ�����Ը����������ֵ��������������ϵ����
post=ones(418,2);
for i=1:418
   if(Age0(i+891)==-1) 
       for j=1:4
       post(i,1)=post(i,1)*(eq(A3(i,j),P1(1,1,j))*P1(1,2,j)+eq(A3(i,j),P1(2,1,j))*P1(2,2,j)+eq(A3(i,j),P1(3,1,j))*P1(3,2,j));
       post(i,2)=post(i,2)*(eq(A3(i,j),P2(1,1,j))*P2(1,2,j)+eq(A3(i,j),P2(2,1,j))*P2(2,2,j)+eq(A3(i,j),P2(3,1,j))*P1(3,2,j));
       end
        post(i,1)= post(i,1)*P;
        post(i,2)=post(i,2)*(1-P);
   else
       prodt1(i)=normpdf(Age0(i+819),mu1,sigma1);
       prodt2(i)=normpdf(Age0(i+819),mu2,sigma2);
       for j=1:4
       post(i,1)=post(i,1)*(eq(A3(i,j),P1(1,1,j))*P1(1,2,j)+eq(A3(i,j),P1(2,1,j))*P1(2,2,j)+eq(A3(i,j),P1(3,1,j))*P1(3,2,j));
       post(i,2)=post(i,2)*(eq(A3(i,j),P2(1,1,j))*P2(1,2,j)+eq(A3(i,j),P2(2,1,j))*P2(2,2,j)+eq(A3(i,j),P2(3,1,j))*P1(3,2,j));
       end
       post(i,1)=prodt1(i)*post(i,1)*P;
       post(i,2)=prodt2(i)*post(i,2)*(1-P);
   end
   
end

for i=1:418
    if(post(i,1)>post(i,2))
        lable(i,1)=0;
    else
        lable(i,1)=1;
    end
end
%--------------------------������ȷ��-----------------------------
% ���Խ�����ݵ������ֱ�ΪSurvived1�� PassengerID1
for r=1:418
    Final(r,1)=xor(lable(r,1),Survived1(r,1));
end
accurary=1-sum(Final)/418
%--------------------------����csv�ļ�-----------------------------
columns={'PassengerId','Survived'};
data=table(PassengerId1,lable,'VariableNames', columns);
writetable(data, 'submission1.csv')