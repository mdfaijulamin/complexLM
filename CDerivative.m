function [Der, DerC] = CDerivative(type, X, Y)
% Derivatives of Complex-valued activation functions.
% Der = w.r.t variable x, and DerC = w.r.t. conjugate of x
% Refer to the paper for notations and explantions.
    switch type
        case 'splitTanh'
            DR = 1 - real(Y).^2;
            DI = 1 - imag(Y).^2;
            Der = 0.5*(DR + DI);
            DerC = 0.5*(DR - DI);
        case 'splitSigm'
            DR = real(Y).*(1 - real(Y));
            DI = imag(Y).*(1 - imag(Y));
            Der = 0.5*(DR + DI);
            DerC = 0.5*(DR - DI);
        case 'linear'
            Der = ones(size(X));
            DerC = zeros(size(X));
        case 'sech'
            Der = -tanh(X).*Y;
            DerC = zeros(size(X));
        case 'tanh'
            Der = sech(X).^2;
            DerC = zeros(size(X));
        case 'sinh'
            Der = cosh(X);
            DerC = zeros(size(X));
        case 'tan'
            Der = sec(X).^2;
            DerC = zeros(size(X));
        case 'sin'
            Der = cos(X);
            DerC = zeros(size(X));
        case 'atan'
            Der = 1./(1 + X.^2);
            DerC = zeros(size(X));
        case 'asin'
            Der = 1./sqrt(1 - X.^2);
            DerC = zeros(size(X));
        case 'acos'
            Der = -1./sqrt(1 - X.^2);
            DerC = zeros(size(X));
        case 'George'
            Denom = (1 + X.*conj(X)).^2;
            Der = 1./Denom;
            DerC = (-X.^2)./Denom;
        case 'atanh'
            Der = 1./(1 - X.^2);
            DerC = zeros(size(X));
        case 'asinh'
            Der = 1./sqrt((1 + X.^2));
            DerC = zeros(size(X));
        otherwise
            error('Unknnown Activation Function');
    end
end