clear all; close all;
%1.Исходные данные для генерации образов M порождающих классов
n=2; M=3;%размерность признакового пространства и число классов
%L - количество компонентов смеси в каждом классе
%dm - параметр, определяющий среднюю степень пересечения компонентов смесей
%romin, romax - границы значений коэффициента корреляции для задания матриц ковариации
L=ones(1,M);%каждый класс порождается одним гауссовским распределением
dm=4;
romin=-0.9;
romax=0.9;
%Веса, математические ожидания, дисперсии и коэффициенты корреляции компонентов смесей
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
% Генерация данных
Ni = 50;
NN = [Ni, Ni, Ni, Ni, Ni];
N = sum(NN); % объемы тестирующих данных
% 1. Исходные данные для генерации образов M порождающих классов (добавлено)
X = gen(n, M, NN, L, ps, mM, D, ro, 0);
Ni = 50;
NN = [Ni, Ni, Ni, Ni, Ni];
N = sum(NN); % объемы тестирующих данных
% Изменения для создания данных XN
Nmi = 0;
Ns = zeros(1, M);
XN = zeros(N, n);
for i = 1:M
    Nma = Nmi + NN(i);
    Ns(i) = Nma;
    XN(Nmi + 1:Nma, :) = X{i}';
    Nmi = Nma;
end
%2. Тестирование алгоритма с метрикой 'sqEuclidean' 
options = statset('Display', 'final', 'MaxIter', 100, 'TolFun', 1e-6); 
[idx, ctrs, sumd] = kmeans(XN, M, 'Distance', 'sqEuclidean', 'replicates', 5, 'Options', options);
figure(1); silhouette(XN, idx); % отображение силуэта
%3. Оценка ошибок, визуализация тестовых данных и ошибочных решений
[ercl, idxn, prM] = erclust(M, NN, idx); % оценка ошибок
disp('Индекс качества кластеризации и частость ошибок'); disp([prM, ercl]);
figure; grid on; hold on;
for i = 1:M
    plot(XN(idxn == i, 1), XN(idxn == i, 2), 'o', 'MarkerSize', 10, 'LineWidth', 1);
end
plot(ctrs(:,1), ctrs(:,2), 'k*', 'MarkerSize', 14, 'LineWidth', 2);