function [ DelayedData ] = TimeDelayEmbedding( Data, t_delay )
% Performs time-delay embedding of data
%   Each example should occupy a row of Data
%   t_delay >= 1 is the number of data samples to embed

N_data = size(Data,1);
x_dim = size(Data,2);

DelayedData = zeros(N_data-t_delay+1,t_delay*x_dim);

for j = 1:N_data-t_delay+1
    for k = 1:t_delay
        low = (k-1)*x_dim+1;
        high = k*x_dim;
        DelayedData(j,low:high) = Data(j+k-1,:);
    end
end

end

