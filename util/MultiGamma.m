function [varargout] = MultiGamma(lambda,n,option)
% Compute multivariate Gamma function, which is used in the probability
% density function of the Wishart and inverse Wishart distributions,
% including Gamma function, logarithm multivariate Gamma function, the
% derivative of logarithm multivariate Gamma function.
%
% Formula:
%     Gamma_n(lambda) = pi^(n(n-1)/4) \prod_(j=1)^n Gamma(lambda+(1-j)/2)
%     log(Gamma_d(lambda)) = n(n-1)/4 log(pi) + \sum_(j=1)^n log(...
%                                Gamma(lambda+(1-j)/2))
%     \partial log(Gamma_d(lambda))/\partial lambda
%          = \sum_(j=1)^n psi(lambda+(1-j)/2)
% Usage:
%     disp('Usage: [GammaY       ] = MultiGamma(lambda,n);')
%     disp('   or: [LogGammaY    ] = MultiGamma(lambda,n,''Log'');')
%     disp('   or: [DerLogGammaY ] = MultiGamma(lambda,n,''DerLog'');')
%     disp('   or: [GammaY,LogGammaY,DerLogGammaY] = MultiGamma(lambda,n,''All'');')
% Input:
%   lambda: the parameter of Gamma function Re(lambda)>1/2*(n-1)
%   n: dimension
%   option: two types: 'Log','LogDer','All'.
% Output:
%   GammaY: multivariate Gamma
%   LogGamma: log multivariate Gamma
%   DerLogGammaY: derivative of log multivariate Gamma
% Remark:
%   The real part of lambda should bigger than 1/2*(n-1) since the
%   definition of multivarivate gamma function
%
% Copyright: Maigca Chen, 2016/05/15
% ?         Last modified 2018/01/19
% Email: sxtpy2010@gmail.com            

if nargin < 2 || nargin >3
    disp('Usage: [GammaY       ] = MultiGamma(lambda,n);')
    disp('   or: [LogGammaY    ] = MultiGamma(lambda,n,''Log'');')
    disp('   or: [DerLogGammaY ] = MultiGamma(lambda,n,''DerLog'');')
    disp('   or: [GammaY,LogGammaY,DerLogGammaY] = MultiGamma(lambda,n,''All'');')
    return
end

s = size(lambda);
if sum(sum(real(lambda)>=0.5*(n-1))) < prod(s)
    error('Re(lambda) must bigger than 0.5*(n-1)')
end


if nargin ==2  % just compute the multivariate gamma function
    if n==1
        GammaY = gamma(lambda);   % use default gamma function
        varargout = {GammaY};
    else
        lambda = reshape(lambda,1,prod(s));
        lambda = bsxfun(@plus,repmat(lambda,n,1),(1-(1:n)')/2);
        
        GammaY = pi^(n*(n-1)/4)*prod(gamma(lambda),1);
        GammaY = reshape(GammaY,s);
        varargout = {GammaY};
    end
end

if nargin == 3
    if n == 1
        switch option
            case 'Log'
                LogGammaY =gammaln(lambda);
                varargout = {LogGammaY};
            case 'DerLog'
                DerLogGammaY =psi(lambda);
                varargout = {DerLogGammaY};
            case 'All'
                GammaY = gamma(lambda);
                LogGammaY =gammaln(lambda);
                DerLogGammaY = psi(lambda);
                varargout={GammaY,LogGammaY,DerLogGammaY};
        end
    else
        
        lambda = reshape(lambda,1,prod(s));
        lambda = bsxfun(@plus,repmat(lambda,n,1),(1-(1:n)')/2);
        
        switch option
            case 'Log'
                LogGammaY = n*(n-1)/4*log(pi)+sum(gammaln(lambda),1);
                LogGammaY = reshape(LogGammaY,s);
                varargout = {LogGammaY};
            case 'DerLog'
                DerLogGammaY =sum(psi(lambda),1);
                DerLogGammaY = reshape(DerLogGammaY,s);
                varargout = {DerLogGammaY};
            case 'All'
                GammaY = pi^(n*(n-1)/4)*prod(gammaln(lambda),1);
                GammaY = reshape(GammaY,s);
                
                LogGammaY = n*(n-1)/4*log(pi)+sum(gammaln(lambda),1);
                LogGammaY = reshape(LogGammaY,s);
                
                DerLogGammaY =sum(psi(lambda),1);
                DerLogGammaY = reshape(DerLogGammaY,s);
                
                varargout={GammaY,LogGammaY,DerLogGammaY};
        end
    end
end

end
