clear; clc; close all

%% Load data

filename_train = 'LinearSystem_Image_HardData_Train.txt';
filename_test = 'LinearSystem_Image_HardData_Test.txt';

RawData_train = importdata(filename_train);
RawData_test = importdata(filename_test);

Len_train = size(RawData_train,1);
Len_test = size(RawData_test,1);

% Takens (time delay) embedding
t_delay = 5;

%% Pre-process data

N_keepdata = 1000;

% remove first column (sample number) of data and normalize
Data_train = RawData_train(1:N_keepdata,2:end);
Data_test = RawData_test(1:N_keepdata,2:end);

% Introduce noise
%Data_train = Data_train + 1e-2 * randn(size(Data_train));

% form time delay embeddings
DelayData_train = TimeDelayEmbedding( Data_train, t_delay );
DelayData_test = TimeDelayEmbedding( Data_test, t_delay );

% normalize data to have unit mean norm
mags = zeros(size(DelayData_train,1),1);
for i = 1:size(DelayData_train,1)
    mags(i) = sqrt(DelayData_train(i,:)*DelayData_train(i,:)');
end
mean_mag = mean(mags);
DelayData_train = DelayData_train/mean_mag;
DelayData_test = DelayData_test/mean_mag;

% form snapshot pairs
N_train = size(DelayData_train,1) - 1;
inds_train = randi(N_train, N_keepdata, 1);
N_train = N_keepdata;
X_train = DelayData_train(inds_train,:)';
Y_train = DelayData_train((inds_train+1),:)';

N_test = size(DelayData_test,1) - 1;
X_test = DelayData_test(1:N_test,:)';
Y_test = DelayData_test(2:N_test+1,:)';

x_dim = size(X_train,1);


%% Generate Kernel Matrices

%kernel = @(x1, x2) (1.0 + x2'*x1).^3;
sig_ker = 2;
kernel = @(x1, x2) exp(-(x2 - x1)'*(x2 - x1)/(2*sig_ker^2));

% form inner product matrices
G_YX = zeros(N_train,N_train);
G_XX = zeros(N_train,N_train);

fprintf('\n *** Generating Kernel Matrices *** \n')
for i = 1:N_train
    for j = 1:N_train
        xi = X_train(:,i);
        xj = X_train(:,j);
        yi = Y_train(:,i);
        
        G_YX(i,j) = kernel(yi, xj);
        
        G_XX(i,j) = kernel(xi, xj);
    end
    
    if mod(i,10)==0
        fprintf('\t Completed %d of %d \n', i, N_train)
    end
end


%% Perform Kernel DMD

% L^1 sparsity parameter
gamma = 1e-3;

K_rank = N_keepdata;

% compute pseudoinverse of G_hat (symmetric matrix)
[V, Sig_sq] = eig(G_XX);
s = sqrt(diag(Sig_sq));
[~, IX] = sort(s, 'descend');
s = s(IX);
V = V(:,IX);

K_rank = min([K_rank, find(s > 1e-6, 1, 'last')])

s_sub = s(1:K_rank);
V_sub = V(:,1:K_rank);

Sig_pinv = diag(1./s_sub);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Koopman approximation matrix in POD basis in feature space 

% % L1 sparsity using non-weighted empirical error 
% Y_mat = V_sub'*G_YX*V_sub*Sig_pinv;
% Ths = gamma ./ (2*s_sub*ones(1,K_rank));
% 
% K_hat = (Y_mat > Ths).*(Y_mat - Ths)./(s_sub*ones(1,K_rank)) +...
%     (Y_mat < -Ths).*(Y_mat + Ths)./(s_sub*ones(1,K_rank));

% L1 sparsity using Weighted empirical error 
Y_mat = V_sub'*G_YX*V_sub;
Ths = gamma ./ (2*(s_sub*s_sub'));

K_hat = (Y_mat > Ths).*(Y_mat - Ths)./(s_sub*s_sub') +...
    (Y_mat < -Ths).*(Y_mat + Ths)./(s_sub*s_sub');

% % L0 sparsity using Weighted empirical error 
% Y_mat = V_sub'*G_YX*V_sub;
% Ths = gamma * ones(K_rank, K_rank);
% 
% K_hat = (Y_mat.^2 > Ths).*Y_mat./(s_sub*s_sub');

% No sparsity
%K_hat = (Sig_pinv*V_sub')*G_YX*(V_sub*Sig_pinv);

fprintf('\n Fraction of Nonzero Elements in K_hat = %.3e \n', ...
    nnz(K_hat(:))/K_rank^2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find Koopman eigenvalues
[W_right, Mu_KDMD, W_left] = eig(K_hat);
mu_KDMD = diag(Mu_KDMD);

% make sure left and right eigenvectors form bi-orthogonal set
scales = diag(W_left'*W_right);
W_left = W_left ./ (ones(K_rank,1)*scales');

% Compute Koopman modes (cols of Xi_KDMD)
Xi_KDMD = X_train*conj(V_sub)*Sig_pinv*conj(W_left);
Xi_KDMD_real = normc(real(Xi_KDMD));

% Function to compute Koopman Eigenfunctions (cols are examples and
% rows are koopman eigenfunctions)
Koop_EigFuns = @(x) W_right.'*Sig_pinv*V_sub.'*conj(KernelVecs(X_train, x, kernel ));

% prediction n-steps into the future
MakePred = @(x,n) Xi_KDMD*diag(mu_KDMD.^n)*Koop_EigFuns(x);


%% Plot Eigenvalues and Modes

% Koopman eigenvalues
figure()
plot(real(mu_KDMD(1:K_rank)), imag(mu_KDMD(1:K_rank)), 'kx',...
    'LineWidth', 2);
hold on
th = linspace(0,2*pi,1000);
plot(cos(th), sin(th), 'r-', 'LineWidth', 1.5);
hold off
grid on
xlim([-1.2, 1.2])
ylim([-1.2, 1.2])
xlabel('Real')
ylabel('Imaginary')
title('Koopman Eigenvalues Computed with KDMD')

% Koopman eigenvalues
figure()
plot(real(log(mu_KDMD(1:K_rank))), imag(log(mu_KDMD(1:K_rank))), 'kx',...
    'LineWidth', 2);
grid on
xlabel('Real')
ylabel('Imaginary')
title('Log Koopman Eigenvalues Computed with KDMD')

% Koopman modes
[~,IX] = sort(abs(mu_KDMD), 'descend');
figure()
PltModes = [1,2,3,5];
for j = 1:length(PltModes);
    
    idx =IX(PltModes(j));
    
    subplot(length(PltModes),1,j)
    %plot(real(Xi_KDMD_real(2*j-1,:)), 'k-', 'LineWidth', 1.5)
    image(reshape(real(Xi_KDMD_real(:, idx)), 11, 11*t_delay),'CDataMapping','scaled')
    colorbar
    title(sprintf('Koopman Mode: \\lambda_{%d} = %.3f + %.3f i', idx,...
        real(mu_KDMD(idx)), imag(mu_KDMD(idx))))
end

% Make some predictions
N_pred = 5;
idx_block = randi(floor(N_test/50));
idx_plus = randi(50-N_pred-1);
idx = 50*(idx_block-1) + idx_plus;
for n = 0:N_pred
    figure()
    pred = real(MakePred(X_test(:,idx),n));
    pp1 = plot(X_test(:,idx+n), 'k-', 'LineWidth', 1.5);
    hold on
    pp2 = plot(pred, 'b-', 'LineWidth', 1.5);
    hold off
    legend([pp1,pp2], {'Ground Truth', 'KDMD'})
    title(sprintf('Prediction and Gound Truth at Test Index %d + %d', idx, n))
end
