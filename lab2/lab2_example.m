clear all
close all

%1.Задание исходных данных
n=2; M=3; % размерность признакового пространства  и число классов
K = 1000; % количество статистических испытаний

m = [0 -1; -4 2; -1 2]';   % мат. ожидания - координаты центров классов (2,-3) и (1,6)
pw = [0.3, 0.3, 0.3];    
np=sum(pw); pw=pw/np;

C = [5 -3; -3 5];   % уменьшенная матрица ковариаций классов
C_ = C^-1;          % обратная ков. матрица

Ks = fix(pw .* K);
Ks(end) = K - sum(Ks(1 : end - 1));
label = {'bo', 'r+', 'k*', 'gx'};       
IMS = [];   
figure; hold on; title('Исходные метки образов');
for i = 1:M 
    ims = repmat(m(:,i), [1, Ks(i)]) + randncor(n, Ks(i), C); 
    if (n == 2)            
        plot(ims(1, :), ims(2, :), label{i}, 'MarkerSize', 10, 'LineWidth', 1);
    elseif (n == 3)
        plot3(ims(1, :), ims(2, :), ims(3, :), label{i}, 'MarkerSize', 10, 'LineWidth', 1);
    end
    IMS = [IMS, ims];   
end

G = zeros(M, n + 1); PIJ = zeros(M); l0_ = zeros(M);
for i = 1:M
    G(i, 1:n) = (C_ * m(:, i))';     
    G(i, n + 1) = -0.5 * m(:, i)' * C_ * m(:, i);
    for j = i + 1:M
        l0_(i, j) = log(2 * pw(j) / pw(i)); 
        h = 0.5 * (m(:, i) - m(:, j))' * C_ * (m(:, i) - m(:, j)); 
        sD = sqrt(2 * h);
        PIJ(i, j) = normcdf(l0_(i, j), h, sD); PIJ(j, i) = 1 - normcdf(l0_(i, j), -h, sD);
    end
    PIJ(i, i) = 1 - sum(PIJ(i, :));
end

figure; hold on; title('Результат классификации образов');
for i = 1:K 
    z = [IMS(:, i); 1];
    u = G * z + log(pw');
    [ui, iai] = max(u);
    if (n == 2)            
        plot(IMS(1, i), IMS(2, i), label{iai}, 'MarkerSize', 10, 'LineWidth', 1);
    elseif (n == 3)
        plot3(IMS(1, i), IMS(2, i), IMS(3, i), label{iai}, 'MarkerSize', 10, 'LineWidth', 1);
    end
end

x = ones(n + 1, 1); Pc_ = zeros(M);
for k = 1:K 
    for i = 1:M 
        [x_, px] = randncor(n, 1, C);        
        x(1:n, 1) = m(:, i) + x_;
        u = G * x + log(pw');
        [ui, iai] = max(u);
        Pc_(i, iai) = Pc_(i, iai) + 1;
    end
end

Pc_ = Pc_ / K;

disp('Теоретическая матрица вероятностей ошибок'); disp(PIJ); 
disp('Экспериментальная матрица вероятностей ошибок'); disp(Pc_);

if n == 2
    D = 1;
    xmin1 = -4 * sqrt(D) + min(m(1, :)); xmax1 = 4 * sqrt(D) + max(m(1, :));
    xmin2 = -4 * sqrt(D) + min(m(2, :)); xmax2 = 4 * sqrt(D) + max(m(2, :));
    x1 = xmin1:0.05:xmax1; x2 = xmin2:0.05:xmax2; 
    figure; hold on; grid on;
    axis([xmin1, xmax1, xmin2, xmax2]);
    
    [X1, X2] = meshgrid(x1, x2);
    x12 = [X1(:), X2(:)]; 
    for i = 1:M
        f2 = mvnpdf(x12, m(:, i)', C);
        f3 = reshape(f2, length(x2), length(x1));
        [Ch, h] = contour(x1, x2, f3, [0.01, 0.5 * max(f3(:))], 'Color', 'b', 'LineWidth', 0.75);
        clabel(Ch, h);
        for j = i + 1:M 
            wij = C_ * (m(:, i) - m(:, j)); wij0 = -0.5 * (m(:, i) + m(:, j))' * C_ * (m(:, i) - m(:, j));
            f4 = wij' * x12' + wij0; f5 = reshape(f4, length(x2), length(x1));
            [~, h_] = contour(x1, x2, f5, [l0_(i, j) + 0.0001, l0_(i, j) + 0.0001], 'Color', 'k', 'LineWidth', 1.25);
        end
    end
    set(gca, 'FontSize', 13);
    title('Области локализации классов и разделяющие границы', 'FontName', 'Courier');
    xlabel('x1', 'FontName', 'Courier'); ylabel('x2', 'FontName', 'Courier'); 
    strv1 = ' pw='; strv2 = num2str(pw, '% G');
    text(xmin1 + 1, xmax2 - 1, [strv1, strv2], 'HorizontalAlignment', 'left', 'BackgroundColor', [.8 .8 .8], 'FontSize', 12);
    legend('wi', 'gij(x)=0'); hold off;
end
