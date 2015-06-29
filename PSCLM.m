% This program implements Wirtinger Calculus based pseudo-Levenberg-Marquardt
% algorithm for complex-valued neural network (A three layer feedfroward)
% 
% Author: Md Faijul Amin {Email: mdfaijulamin@yahoo.com}
% 
% Ref: M.F. Amin, M.I. Amin, A.Y.H. Al-Nuaimi, and K. Murase. 2011. 
% Wirtinger Calculus based Gradient Descent and Levenberg-Marquardt
% Learning Algorithms in Complex-Valued Neural Networks. B.-L. Lu, L.
% Zhang, and J. Kwok (Eds.): ICONIP 2011, Part I, LNCS 7062, pp. 550-559.

clear all; clc;
% Load data
load data;
targetData = targetData/max(abs(targetData));
nPatterns = size(inputData, 2);
nTrPatterns = 8;

% Number of neurons at different layers
nInp = size(inputData, 1);
nHid = 4;
nOut = size(targetData, 1);

rand('seed', 87521);

% Activation function types
typeHid = 'splitSigm'; typeOut = 'splitSigm';

MAX_EPOCH = 200;    % Maximum number of epoch
MU_MAX = 1e10;
% Construct the network and intialize the weight parameters
Wih = (rand(nHid, nInp)-0.5) + 1j*(rand(nHid, nInp) - 0.5);
bHid = (rand(nHid, 1)-0.5) + 1j*(rand(nHid, 1)-0.5);
Who = (rand(nOut, nHid)-0.5) + 1j*(rand(nOut, nHid)-0.5);
bOut = (rand(nOut, 1)-0.5) + 1j*(rand(nOut, 1)-0.5);

% Sotorage for neruons net-input and output; all are supposed to be column
% vectors
vHid = []; yHid = []; vOut = []; yOut = [];

% History of training progress
trErr = [];
mu = 0.01;

% Forward calculation
X = inputData(:, 1:nTrPatterns);
T = targetData(:, 1:nTrPatterns);
VHid = Wih*X + repmat(bHid, 1, nTrPatterns);
YHid = CActFunc(typeHid, VHid);
VOut = Who*YHid + repmat(bOut, 1, nTrPatterns);
YOut = CActFunc(typeOut, VOut);
E = T - YOut;
e = E(1:end).';
err = real(e'*e)/nTrPatterns; 
trErr = [trErr; err];
tstart = tic;
for epoch = 1:MAX_EPOCH
    % Construct appropriate Jacobian matrix
    % J = [J1, J2, J3, J4]
    % Each submatrix Ji corresponds to subset of parameters, weights and
    % biases. Conjugate Jacobians are stored in JC = [J1C, J2C, J3C, J4C]
    
    [DYOut, DYOutC] = CDerivative(typeOut, VOut, YOut);
    [DYHid, DYHidC] = CDerivative(typeHid, VHid, YHid);
    
    J1 = kron(DYOut.', ones(nOut,1))...
        .*kron(ones(nTrPatterns, 1), eye(nOut));
    J1C = kron(DYOutC.', ones(nOut,1))...
        .*kron(ones(nTrPatterns, 1), eye(nOut));
    J2 = kron(J1, ones(1, nHid))...
        .*kron(ones(1,nOut), kron(YHid.', ones(nOut,1)));
    J2C = kron(J1C, ones(1, nHid))...
        .*kron(ones(1,nOut), kron(YHid', ones(nOut,1)));
    J3 = J1*Who.*kron(DYHid.', ones(nOut,1)) +...
            J1C*conj(Who).*kron(DYHidC', ones(nOut,1));
    J3C = J1*Who.*kron(DYHidC.', ones(nOut,1)) +...
            J1C*conj(Who).*kron(DYHid', ones(nOut,1));            
    J4 = kron(J3, ones(1, nInp))...
        .*kron(ones(1,nHid), kron(X.', ones(nOut,1)));
    J4C = kron(J3C, ones(1, nInp))...
        .*kron(ones(1,nHid), kron(X', ones(nOut,1)));
    J = [J1, J2, J3, J4];
    JC = [J1C, J2C, J3C, J4C];    
   
   while (mu<=MU_MAX)
        % Pseudo-Gauss-Newton Method
        m = size(J,2);
        H = (J'*J + conj(JC'*JC) + mu*eye(m));
        b = J'*e + conj(JC'*e);
        deltaZ = H\b;
        
        % Update Weights and biases
        bOut1 = bOut + deltaZ(1:nOut);
        Who1 = Who + reshape(deltaZ(nOut+1:nOut*(1+nHid)), nHid, nOut).';
        bHid1 = bHid + deltaZ(nOut*(1+nHid)+1: nOut*(1+nHid)+nHid);
        Wih1 = Wih + reshape(deltaZ(nOut*(1+nHid)+nHid+1: end), nInp, nHid).';
        
        % Check the update
        VHid = Wih1*X + repmat(bHid1, 1, nTrPatterns);
        YHid = CActFunc(typeHid, VHid);
        VOut = Who1*YHid + repmat(bOut1, 1, nTrPatterns);
        YOut = CActFunc(typeOut, VOut);
        E = T - YOut;
        e1 = E(1:end).';
        err1 = real(e1'*e1)/nTrPatterns; 
        if err1<err
            bOut = bOut1;
            Who = Who1;
            bHid = bHid1;
            Wih = Wih1;
            e = e1;
            err = err1;
            mu = mu/10;
            if mu<1e-20
                mu = 1e-20;
            end
            break;
        end
        mu = mu*10;        
   end
   if mu>MU_MAX
       break;
   end
   trErr = [trErr; err];
   if trErr(end)<0.0001
       break;
   end
end
toc(tstart)

% See training error history
trErr

