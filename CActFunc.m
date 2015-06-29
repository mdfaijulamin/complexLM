function [Y] = CActFunc(type, X)
% Complex-valued activation functions. 
    switch type
        case 'splitTanh'
            R = tanh(real(X));
            I = tanh(imag(X));
            Y = R + 1j*I;
        case 'splitSigm'
            R = 1./(1 + exp(-real(X)));
            I = 1./(1 + exp(-imag(X)));
            Y = R + 1j*I;
        case 'linear'            
            Y = X;
        case 'sech'
            Y = sech(X);
        case 'tanh'
            Y = tanh(X);
        case 'sinh'
            Y = sinh(X);
        case 'tan'
            Y = tan(X);
        case 'sin'
            Y = sin(X);
        case 'atan'
            Y = atan(X);
        case 'asin'
            Y = asin(X);
        case 'acos'
            Y = acos(X);
        case 'George'
            Y = X./(1 + X.*conj(X));
        case 'atanh'
            Y = atanh(X);
        case 'asinh'
            Y = asinh(X);
        otherwise
            error('Unknown Activation Function');
end