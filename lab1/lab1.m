%%% Вычисление выборочных характеристик гауссовской случайной величины (ГСВ)
clear all
close all

%% 1. Задание исходных данных
% Параметры генерации
n = 10; % число реализаций равномерной случайной величины для генерации одной реализации биномиальной СВ
p = 0.5; % вероятность положительного исхода в каждой реализации
N = 1000; % число реализаций

%% 2. Вычисление значений статистических характеристик ГСВ
m = n * p; % мат. ожидание

%% 3. Генерация реализаций случайной величины
% Генерация реализаций стандартной РСВ
alf = rand(n, N); % матрица из N столбцов по n элементов
% Генерация реализаций ГСВ
x = sum(alf<=p); % сумма по столбцам матрицы alf<=p

%% 4. Вычисление выборочных характеристик
M = mean(x); % выборочное среднее
D = var(x);  % выборочная дисперсия
% Вывод значений теоретических и выборочных характеристик
disp('Среднее значение (теоретическое)'); 
disp(m);
disp('Среднее значение (выборочное)');
disp(M);

ms = zeros(1, N);
for k = 1 :N
    ms(k) = mean(x(1 : k)); % среднее первых k реализаций
end

figure; hold on; % создание графического окна
plot(1 : N, ms); % отображение зависимости выборочных средних от числа реализаций СВ
plot(1 : N, m * ones(1, N), 'g'); % отображение значения мат. ожидания
title('Выборочное среднее от числа реализаций'); % заголовок
legend(['Выборочное математическое ожидание = ' num2str(M)], ...
       ['Теоретическое математическое ожидание = ' num2str(m)]); % легенда

