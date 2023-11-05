"DiscriminativeKalmanFilter.m";

% data source
cd(fileparts(mfilename('fullpath')));
z = dlmread('../data/z.csv');
x = dlmread('../data/x.csv');

% dimensions of latent states and observations, respectively
dz = size(z,2);
dx = size(x,2);

% training data
train_idx = 1:5000;
n_train = length(train_idx);
x_train = x(train_idx,:);
z_train = z(train_idx,:);

% test data
test_idx = 5000+1:6e3;
n_test = length(test_idx);
x_test = x(test_idx,:);
z_test = z(test_idx,:);

% learn state model parameters A & Gamma from Eq. (2.1b)
A0 = z_train(2:end,:)'/z_train(1:end-1,:)';
resids = z_train(2:end,:)'-A0*z_train(1:end-1,:)';
Gamma0 = resids*resids'/n_train;

% learn f & Q with ordinary least squares
H0 = z_train'/x_train';
fx = @(x) H0*x';
resids = z_train'-H0*x_train';
Qx = @(x) resids*resids'/n_train;

% initialize DKF using learned parameters
f0 = fx(x_test(1,:));
Q0 = Qx(x_test(1,:));
DKF = DiscriminativeKalmanFilter(A0,Gamma0,cov(z_train),fx,Qx,f0,Q0);

% perform filtering
z_preds = zeros(n_test,dz);
z_preds(1,:) = f0;
for i = 2:n_test
    z_preds(i,:) = DKF.predict(x_test(i,:));
    z_preds(i,:) = fx(x_test(i,:));
end

% handle output
disp("normalized rmse")
disp(sqrt(mean((z_test(:)-z_preds(:)).^2))/ sqrt(mean((z_test(:)).^2)));
