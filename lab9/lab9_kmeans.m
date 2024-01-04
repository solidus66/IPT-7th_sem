clear all; close all;
%1.�������� ������ ��� ��������� ������� M ����������� �������
n=2; M=3;%����������� ������������ ������������ � ����� �������
%L - ���������� ����������� ����� � ������ ������
%dm - ��������, ������������ ������� ������� ����������� ����������� ������
%romin, romax - ������� �������� ������������ ���������� ��� ������� ������ ����������
L=ones(1,M);%������ ����� ����������� ����� ����������� ��������������
dm=4;
romin=-0.9;
romax=0.9;
%����, �������������� ��������, ��������� � ������������ ���������� ����������� ������
ps=cell(1,M);
mM=cell(1,M);
D=cell(1,M);
ro=cell(1,M);
for i=1:M
    ps{i}=ones(1,L(i))/L(i);
    D{i}=ones(1,L(i));
    ro{i}=romin+(romax-romin)*rand(1,L(i));
end
mM{1}=[0;0]; mM{2}=[0;dm]; mM{3}=[dm;0];
% ��������� ������
Ni = 50;
NN = [Ni, Ni, Ni, Ni, Ni];
N = sum(NN); % ������ ����������� ������
% 1. �������� ������ ��� ��������� ������� M ����������� ������� (���������)
X = gen(n, M, NN, L, ps, mM, D, ro, 0);
Ni = 50;
NN = [Ni, Ni, Ni, Ni, Ni];
N = sum(NN); % ������ ����������� ������
% ��������� ��� �������� ������ XN
Nmi = 0;
Ns = zeros(1, M);
XN = zeros(N, n);
for i = 1:M
    Nma = Nmi + NN(i);
    Ns(i) = Nma;
    XN(Nmi + 1:Nma, :) = X{i}';
    Nmi = Nma;
end
%2. ������������ ��������� � �������� 'sqEuclidean' 
options = statset('Display', 'final', 'MaxIter', 100, 'TolFun', 1e-6); 
[idx, ctrs, sumd] = kmeans(XN, M, 'Distance', 'sqEuclidean', 'replicates', 5, 'Options', options);
figure(1); silhouette(XN, idx); % ����������� �������
%3. ������ ������, ������������ �������� ������ � ��������� �������
[ercl, idxn, prM] = erclust(M, NN, idx); % ������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM, ercl]);
figure; grid on; hold on;
for i = 1:M
    plot(XN(idxn == i, 1), XN(idxn == i, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1);
end
plot(ctrs(:,1), ctrs(:,2), 'k*', 'MarkerSize', 14, 'LineWidth', 2);