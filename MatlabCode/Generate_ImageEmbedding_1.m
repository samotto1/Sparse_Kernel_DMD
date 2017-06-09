clear; clc; close all

%% Define Data Sample
%   - Data consists of greyscale images with fixed spatial modes whose
%   coefficients evolve in time according to a specified ordinary
%   differential equation.
%   - Vectorized images form the rows of the output data matrix

N_Samples = 2000; % total number of training points

% time interval for random initialization
Dt_randinit = 50;

% Image dimensions
Width = 11;
Height = 11; 

filname = 'LinearSystem_Image_Data1_Test.txt';
fileID = fopen(filname, 'w'); % write
%fileID = fopen(filname, 'a'); % append

%% Define the underlying linear dynamical system

% Choose eigenvalues
golden = (1+sqrt(5))/2;
D_theta = [pi/5, pi/(5*golden)];
Decay = [1.0, 1.0];

% form system (state transition matrix)
N_states = 2*length(D_theta);
A = zeros(N_states,N_states);
for jj = 1:length(D_theta)
    j = 2*jj - 1;
    A(j,j) = Decay(jj)*cos(D_theta(jj));
    A(j,j+1) = Decay(jj)*sin(D_theta(jj));
    A(j+1,j) = -Decay(jj)*sin(D_theta(jj));
    A(j+1,j+1) = Decay(jj)*cos(D_theta(jj));
end
    
% simulate the system with randomly chosen initial conditions every
% Dt_randinit steps
X = zeros(N_states, N_Samples);
for t = 1:N_Samples-1
    
    % random initialization
    if mod(t-1, Dt_randinit) == 0
        X(:,t) = randn(N_states,1);
    end
    
    % update
    X(:,t+1) = A*X(:,t);
    
end

%% Define nonlinear image embedding

% define a gaussian blip
blip = @(x,x0,sig) exp(-(x-x0)'*(x-x0)/sig^2);

% define an image
xx = linspace(-Width/2, Width/2, Width);
yy = linspace(-Height/2, Height/2, Height);
[X_grid, Y_grid] = meshgrid(xx,yy);


Z_Data = zeros(N_Samples, Width*Height);

fprintf('\n ***Writing Data to File*** \n')
figure()
for n = 1:N_Samples
    
    fprintf(fileID, '%d\t', n);
    
    if mod(n,100) == 0
        fprintf('\n Generated %d of %d \n', n, N_Samples);
    end
    
    % parse image using blips
    Snapshot = zeros(Height,Width);
    for i=1:Height
        for j=1:Width
            loc = [X_grid(i,j); Y_grid(i,j)];
            
            Snapshot(i,j) = Snapshot(i,j) + ...
                5*X(1,n)*blip(loc, [3*X(1,n); (3*X(1,n))^2/5-2.5], 2) + ...
                X(3,n)^3*sin(2*pi/4*loc(1));       
        end
    end
    
%     % draw image
%     image(Snapshot,'CDataMapping','scaled')
%     set(gca, 'CLim', [-6,6])
%     colormap gray
%     title(sprintf('Snapshot %d of %d', n, N_Samples))
%     colorbar
%     drawnow
    
    SnapVec = reshape(Snapshot, [Height*Width,1]);
    fprintf(fileID, '%f\t', SnapVec);
    
    fprintf(fileID, '\r\n');
end

fclose(fileID);   