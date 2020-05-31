% Compute MMD distance using Matlab
% Inputs X and Y are all matrices (n_sample * dim)

function [res] = mmd_matlab(X, Y, kernel_type, gamma)
    if (nargin < 3)
        disp('Please input the kernel_type!');
        res = -9999;
        return;
    end
    switch kernel_type
        case 'linear'
            res = mmd_linear(X, Y);
        case 'rbf'
            if (nargin < 4)
                gamma = 1;
            end
            res = mmd_rbf(X, Y, gamma);
    end
end

function [res] = mmd_linear(X, Y)
    delta = mean(X) - mean(Y);
    res = delta * delta';
end

function [res] = mmd_rbf(X, Y, gamma)
    ker = 'rbf';
    XX = kernel(ker, X', [], gamma);
    YY = kernel(ker, Y', [], gamma);
    XY = kernel(ker, X', Y', gamma);
    res = mean(mean(XX)) + mean(mean(YY)) - 2 * mean(mean(XY));
end

function K = kernel(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end