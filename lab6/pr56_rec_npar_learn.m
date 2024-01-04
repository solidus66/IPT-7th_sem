%���� pr56_rec_npar_learn. �������� ���������� ������������� �� ������ ������
%���������� ������������� �������� ������� � k - ��������� �������
clear all;
close all;
%1.������� �������� ������
n=2;M=2;%����������� ������������ ������������ � ����� �������
H=1;%���������� �������������� ��������� �������� ��������
K=1000;%���������� �������������� ��������� ��������� �������������
pw=ones(1,M)/M;%��������� ����������� ������� (�������� �� ���������)
%�������� ������ ��� ��������� ��������� ������� 
%dm - ��������, ������������ ������� ������������������ ����������� ������
%DM - ��������, ������������ ����� �������� ����������� ������� �� ����
%L - ���������� ����������� � ����� ������� ������ 
L=[2,2];dm=2; DM=1;%��� ���.5.13
%L=[2,3]; dm=2; DM=0;%��� ���.5.14
ps=cell(1,M); mM=cell(1,M); D=cell(1,M); ro=cell(1,M);
%����, �������������� ��������, ��������� � ������������ ���������� ����������� ������
for i=1:M, 
    ps{i}=ones(1,L(i))/L(i); mM{i}=zeros(n,L(i))+(i-1)*DM; D{i}=ones(1,L(i));%�� ���������
    if n==2, ro{i}=2*rand(1,L(i))-1; else ro{i}=zeros(1,L(i)); end;%�� ���������
end;
mM{1}=[[0;0],[dm;dm]]; mM{2}=[[dm;0],[0;dm]]+DM;%��� ���.5.13 ��� n=2
%mM{1}=[[0;0],[dm;dm]]; mM{2}=[[dm;0],[0;dm],[-dm;-dm]]+DM; %��� ���.5.14 ��� n=2
kl_kernel=11; r=0.5; gm=0.25;%��������� ������ (��.�������� ������� vkernel,vknn)

%2.��������� ��������� ������ � ����� � ���������� ������� �������
Nn=[10,20,30,40,50,100,150,200,250]; 
ln=length(Nn);
Esth1=zeros(1,ln);
Esth2=Esth1;
Esex1=Esth1;
Esex2=Esth1;
for nn=1:ln,%���� � ���������� ������ ��������� �������
    NN=Nn(nn)*ones(1,M); N=Nn(nn);
    h_N=N^(-r/n);%������� ���� �������
    k=2*round(N^gm)+1;%k - ����� ��������� �������
    for h=1:H,%���� �������������� ��������� �������� �������� 
        XN=gen(n,M,NN,L,ps,mM,D,ro,h);%��������� ��������� �������
        
        %3.����������� ������������ ������ ������� ����������� ��������
        Pc1=zeros(M); Pc2=zeros(M);%������� ������
        p1_=zeros(M,1); p2_=zeros(M,1);
        for i=1:M,%���������� ������ ����������� ��������
             XNi=XN{i}; XNi_=zeros(n,N-1);
             indi=[1:i-1,i+1:M];
             for j=1:N,
                x=XNi(:,j); indj=[1:j-1,j+1:N];%������� ��������� ������ i-�� ������
                XNi_(:,1:j-1)=XNi(:,1:j-1); XNi_(:,j:end)=XNi(:,j+1:end);
                p1_(i)=vkernel(x,XNi_,h_N,kl_kernel);%������ �������
                p2_(i)=vknn(x,XNi_,k);%������ k - ��������� �������
                for t=1:M-1,
                     ij=indi(t);
                     p1_(ij)=vkernel(x,XN{ij},h_N,kl_kernel);
                     p2_(ij)=vknn(x,XN{ij},k);
                end;
                [ui1,iai1]=max(p1_);[ui2,iai2]=max(p2_);  %����������� ����������
                Pc1(i,iai1)=Pc1(i,iai1)+1;Pc2(i,iai2)=Pc2(i,iai2)+1;%�������� ����������
             end;
             Pc1(i,:)=Pc1(i,:)/N; Pc2(i,:)=Pc2(i,:)/N;
        end;
        Esth1(nn)=Esth1(nn)+(1-sum(pw.*diag(Pc1)')); %��������� ������
        Esth2(nn)=Esth2(nn)+(1-sum(pw.*diag(Pc2)')); %��������� ������
        
        %4.������������ ���������� ������� �������������� ���������
        Pc1_=zeros(M); Pc2_=zeros(M);%����������������� ������� ������
        X=gen(n,M,K*ones(1,M),L,ps,mM,D,ro,0);%��������� ����������� �������
        p1x=zeros(M,K);p2x=zeros(M,K);
        for i=1:M,%���� �� �������
             xi=X{i};
             for j=1:M,
                 p1x(j,:)=vkernel(xi,XN{j},h_N,kl_kernel);%������ �������
                 p2x(j,:)=vknn(xi,XN{j},k);%������ k - ��������� �������
             end;
             [mui1,mai1]=max(p1x); [mui2,mai2]=max(p2x); 
             ni1=find(mai1==i); ni2=find(mai2==i);
             Pc1_(i,i)=length(ni1)/K; Pc2_(i,i)=length(ni2)/K;%�������� ����������
        end;
        Esex1(nn)=Esex1(nn)+(1-sum(pw.*diag(Pc1_)')); %��������� ������
        Esex2(nn)=Esex2(nn)+(1-sum(pw.*diag(Pc2_)')); %��������� ������
   end;%�� h
end;%�� nn
Esth1=Esth1/H;
Esth2=Esth2/H;
Esex1=Esex1/H;
Esex2=Esex2/H;

%5.������������ ������������ ������������ ������
figure; grid on; hold on;
ms=max([Esth1,Esth2,Esex1,Esex2]);
axis([Nn(1),Nn(ln),0,ms+0.001]);%��������� ������ ���� ������� �� ����
p=plot(Nn,Esth1,'-b',Nn,Esth2,'-r',Nn,Esex1,'--ok',Nn,Esex2,'--^k');set(p,'LineWidth',1.0);
title( 'C�������� ����������� ������','FontName','Courier','FontSize',14);
xlabel('N','FontName','Courier','FontSize',14);ylabel('Es','FontName','Courier','FontSize',14);
strv1=' pw='; strv2=num2str(pw,'% G'); strv3=' n='; strv4= num2str(n);
strv5=' L='; strv6= num2str(L,'% G');
strv7=' dm='; strv8=num2str(dm); strv9=' DM='; strv10= num2str(DM);
strv11=' H='; strv12= num2str(H);
text(Nn(3)+1,0.5*ms, [strv1,strv2,strv3,strv4,strv5,strv6,strv7,strv8,strv9,strv10,strv11,strv12],...
    'HorizontalAlignment','left','BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('Esth1 ','Esth2','Esex1','Esex2',1); hold off;







