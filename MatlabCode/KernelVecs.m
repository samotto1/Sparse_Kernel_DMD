function [ Kernel_Vecs ] = KernelVecs(X_train, X, kernel )
% Evaluate kernel with examples and training data
%   evaluate kernel for examples in cols of X

N_examples = size(X,2);
N_train = size(X_train,2);
x_dim = size(X_train,1);

Kernel_Vecs = zeros(N_train, N_examples);
for i = 1:N_train
    for j = 1:N_examples
        xi = X_train(:,i);
        xj = X(:,j);
        Kernel_Vecs(i,j) = kernel(xi, xj);
    end
end


end

