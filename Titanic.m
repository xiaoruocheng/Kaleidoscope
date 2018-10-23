clc
%--------------------------------数据转换---------------------------------
%---------------------将train.csv与test.csv的数据续接在一起
str=['male'];
for i=1:1309
    Sex0(i,1)=strcmp(Sex(i,:),str)    % 性别 男-0 女-1 转换
    if(Sex0(i,:))
    Sex1(i,1)=0;
    else
    Sex1(i,1)=1;
    end
    
str2=char(Embarked(i,1));   %港口转换 C Q S- 1 2 3
switch(str2)
    case'C'
        Embarked1(i,1)=1;
    case'Q' 
        Embarked1(i,1)=2;
	case'S'
        Embarked1(i,1)=3;
end

count(i,1)=SibSp(i,1)+Parch(i,1);  %是否有亲友同行 有1 无0
if count(i,1)>0
    count(i,1)=1;
end
end
% 无年龄者用-1填充
Age0=Age;
Age0(isnan(Age0(:,1)))=[-1];
%---------------------------------------分类-----------------------------------------
A=[ Pclass(1:891,1), Sex1(1:891,1), count(1:891,1), Embarked1(1:891,1), Age0(1:891,1)];
% 将不同类分开 A1 A2分别为死亡和存活群体 A3为测试群体，M1 M2分别为两群体中有年龄的个体 
%假设服从正态分布 ，mu1 mu2 sigma1 sigma2分别为两类的均值和方差
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
%-------------------------计算各项离散概率值-----------------------------------------       
P=length(A1)/length(A); %计算先验概率 P_alive为存活概率
P_alive=1-P;

P1=zeros(3,2,4); %求训练数据各变量概率 Pclass Sex1 count Emparked1 ，P1 P2分别表示死亡和存活组概率
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
      
 
% 计算后验概率 post为后验概率 prodt为似然函数 证据因子相同 对有年龄的人额外加入年龄的似然函数乘积
% 运用equal函数将每个测试个体各特征的值和类条件概率联系起来
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
%--------------------------计算正确率-----------------------------
% 测试结果数据导入名分别为Survived1和 PassengerID1
for r=1:418
    Final(r,1)=xor(lable(r,1),Survived1(r,1));
end
accurary=1-sum(Final)/418
%--------------------------导出csv文件-----------------------------
columns={'PassengerId','Survived'};
data=table(PassengerId1,lable,'VariableNames', columns);
writetable(data, 'submission1.csv')