"DiscriminativeKalmanFilter.m";

% data source
addpath(genpath(fileparts(mfilename('fullpath'))));
load('../data/exampleData.mat');
vel = procd(1).velocities;
spk = procd(1).spikes;
z = vel(2:end,:);
x = spk(1:end-1,:);

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

% learn f() as a neural network
nn = feedforwardnet(10,'trainbr');
nn = configure(nn,x_train',z_train');
nn = init(nn);
nn.divideParam.trainRatio = 0.9;
nn.divideParam.valRatio = 0.;
nn.divideParam.testRatio = 0.1;
[nn,tr] = train(nn,x_train',z_train');
fx = @(x) nn(x');

% learn Q() as a constant on held-out training data
n_train_covariance = length(tr.testInd);
x_train_covariance = x_train(tr.testInd,:)';
z_train_covariance = z_train(tr.testInd,:)';
z_train_preds = nn(x_train_covariance)';
cov_est = zeros(dz,dz);
for i = 1:n_train_covariance
    err_i = z_train_preds(i,:) - z_train_covariance(:,i)';
    cov_est = cov_est + err_i' * err_i / n_train_covariance;
end
Qx = @(x) cov_est;

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
disp(sqrt(mean((z_test-z_preds).^2,'all'))/ sqrt(mean((z_test).^2,'all')));



