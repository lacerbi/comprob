function [cp,P0] = comprob(type,varargin)
%COMPROB Estimate compatibility probability of measurements across experiments.
%   CP = COMPROB('samples',X_1,...X_M) returns the overall compatibility 
%   probability CP across experiments X_1,...,X_M, for N subjects. 
%   X_i{j} for 1 <= i <= M and 1 <= j <= N is an array of samples from the 
%   posterior representing uncertainty about a given parameter for the i-th 
%   experiment and j-th subject.
%   CP represents the posterior probability that the measurements are on
%   average mutually compatible across experiments, for each subject, above
%   and beyond chance.
%
%   CP = COMPROB('pdf',X_1,...X_M) returns the overall compatibility 
%   probability CP, where the X_i{j} is the (unnormalized) pdf representing
%   the posterior over a given parameter for the i-th experiment and j-th
%   subject. The pdf is passed as an array of pdf values defined on an
%   equispaced grid, which is assumed to be the same for all subjects and
%   experiments.
%
%   CP = COMPROB(...,'Prior',PRIOR) specifies the adopted empirical Bayes 
%   prior. Allowed values for PRIOR are:
%      'Cauchy'         Truncated Cauchy prior (default)
%      'uniform'        Uniform prior
%
%   [CP,P0] = COMPROB(...) also returns the posterior probability of the
%   null hypothesis (that parameters are compatible across experiments)
%   for each individual subject.
%
%   Reference:
%   [1] Acerbi, L., Dokka, K., Angelaki, D. E. & Ma, W. J. (2017). Bayesian 
%   comparison of explicit and implicit causal inference strategies in 
%   multisensory heading perception. bioRxiv.

%   Author: Luigi Acerbi, 2017
%   e-mail: luigi.acerbi@{gmail.com,nyu.edu}
%   URL: http://luigiacerbi.com
%   Release date: June 13, 2017

% Parse input for OPTIONS
M = find(cellfun(@ischar,varargin),1) - 1;
if isempty(M) && isstruct(varargin{end})
    M = numel(varargin) - 1;    % OPTIONS struct passed as last argument
elseif isempty(M)
    M = numel(varargin);        % No additional options
end

% Default options
options.TolP = 1e-8;        % Minimum probability
options.Prior = 'Cauchy';   % Empirical Bayes prior ('Cauchy' or 'Uniform')
options.MeshSize = 2^12;    % Mesh size for KDE and numerical integration ('Sampling' only)

% Parse additional inputs
if numel(varargin) > M
    options = parseoptions(options,varargin{M+1:end});
end

if iscell(varargin{1})
    N = numel(varargin{1});
else
    N = size(varargin{1},1);
end

% if S > 1e3; warning(['Detected S = ' num2str(S) ' subjects. This ']); end

nSamples = zeros(N,M);            
lb = zeros(N,M);
ub = zeros(N,M);

switch lower(type(1))
    case 's'
        % Distributions provided as samples; compute KDE
        for i = 1:M
            for j = 1:N
                if iscell(varargin{i})
                    X{i,j} = varargin{i}{j};
                else
                    X{i,j} = varargin{i}(j,:);                    
                end
                nSamples(i,j) = numel(X{i,j});
                lb(i,j) = min(X{i,j});
                ub(i,j) = max(X{i,j});
            end
        end
        LB = min(lb(:));
        UB = max(ub(:));
        Nx = options.MeshSize;        
        xx = linspace(LB,UB,Nx);

        b = zeros(N,M);
        for i = 1:M
            pdf{i} = zeros(N,Nx);
            for j = 1:N
                [b(i,j),kdepdf,xmesh] = kde(X{i,j},Nx,LB,UB);
                if numel(xmesh) ~= Nx
                    pdf{i}(j,:) = interp1(xmesh, kdepdf, xx, 'linear');
                else
                    pdf{i}(j,:) = kdepdf;
                end
            end
        end
    case 'p'
        % Pdfs already provided
        if iscell(varargin{1})
            Nx = numel(varargin{1}{1});
        else
            Nx = size(varargin{1},2);
        end
        for i = 1:M
            pdf{i} = zeros(N,Nx);
            for j = 1:N
                if iscell(varargin{i})
                    currpdf = varargin{i}{j}(:)';                    
                else
                    currpdf = varargin{i}(j,:);
                end
                if size(currpdf,2) ~= Nx
                    error('All input PDFs should be defined on the same grid.');
                end
                pdf{i}(j,:) = currpdf;
            end            
        end
    otherwise
        error('TYPE should be ''(p)df'' or ''(s)amples'' for the chosen input type.');
end

% Normalize pdfs
for i = 1:M
    pdf{i} = bsxfun(@rdivide, pdf{i}, sum(pdf{i},2));
end

% Compute overlap metric
H0 = ones(N,Nx);
H1 = ones(N,1);

meanpostpdf = zeros(1,Nx);
for i = 1:M
    meanpostpdf = meanpostpdf + mean(pdf{i},1)/M;
end
meanpostcdf = cumsum(meanpostpdf);
meanpostcdf = meanpostcdf / meanpostcdf(end);

% plot(meanpostpdf)        
Pleft = find(meanpostcdf >= options.TolP,1,'first');
Pright = find(meanpostcdf <= 1-options.TolP,1,'last');

% Empirical Bayes prior
switch lower(options.Prior(1))
    case 'c'
        % Fit Cauchy to mean posterior
        xx = 1:Nx;
        x0init = find(meanpostcdf >= 0.5,1);
        gammainit = log(0.5*(find(meanpostcdf >= 0.75,1) - find(meanpostcdf >= 0.25,1)));
        theta = fminsearch(@klcauchy,[x0init,gammainit]);
        x0 = theta(1);          % Location parameter
        gamma = exp(theta(2));  % Scale parameter
        % Normalization factor for truncated Cauchy
        nf = 1/pi*(atan((Pright-x0)/gamma) - atan((Pleft-x0)/gamma));
        tcauchypdf = repmat(1./(pi*gamma*nf) .* (gamma^2./(gamma^2 + (xx - x0).^2)), ...
            [N 1]);
        tcauchypdf(:,(xx < Pleft) | (xx > Pright)) = 0;

        for i = 1:M; H0 = H0 .* pdf{i}; end
        H0 = sum(H0 .* tcauchypdf, 2);    
        for i = 1:M; H1 = H1 .* sum(pdf{i} .* tcauchypdf, 2); end

        % figure;
        % plot(meanpostpdf); hold on;
        % plot(tcauchypdf(1,:));
        
    case 'u'
        Np = Pright - Pleft + 1;    
        for i = 1:M; H0 = H0 .* pdf{i}; end
        H0 = sum(H0,2) / Np;
        H1 = ones(N,1) / Np.^M;
        
    otherwise
        error('Available empirical priors in OPTIONS.Prior are (C)auchy and (U)niform.');
end

y = [log(H0), log(H1)];
[~,~,~,cp] = spm_BMS_fast(y);


Nf = H0 + H1;
P0 = H0 ./ Nf;

    function kl = klcauchy(theta)
        %KLCAUCHY KL divergence for truncated Cauchy distribution.        
        x0 = theta(1);
        gamma = exp(theta(2));        
        nf = 1/pi*(atan((Pright-x0)/gamma) - atan((Pleft-x0)/gamma));
        logcauchy = -log(pi*gamma*nf*(1+((xx-x0)/gamma).^2));        
        kl = -sum(meanpostpdf .* logcauchy);
    end
end

%--------------------------------------------------------------------------

function options = parseoptions(options,varargin)
%PARSEOPTIONS Parse options either as struct or variable arguments in name/value format.
%   OPTIONS = PARSEOPTIONS(OPTIONS,'PROPERTY1',VALUE1,'PROPERTY2',VALUE2,...)
%   sets the fields propertyX in default OPTIONS structure to valueX. Input 
%   field names are not case sensitive. 
%
%   OPTIONS = PARSEOPTIONS(OPTIONS,NEWOPTS) assigns fields in struct NEWOPTS
%   to OPTIONS. Matching fields are not case sensitive for OPTIONS.
%
%   OPTIONS = PARSEOPTIONS(OPTIONS,NEWOPTS,'PROPERTY1',VALUE1,'PROPERTY2',VALUE2,...) 
%   first assigns values from struct NEWOPTS, and then name/value pairs.
%

%   Author: Luigi Acerbi
%   Email:  luigi.acerbi@gmail.com
%   Date:   Sep/08/2016

if nargin < 1; help parseoptions; return; end

if isempty(options)
    error('parseOptions:emptyOptions','Default OPTIONS struct should be nonempty.');
end

if isempty(varargin)
    return;
end

deff = fields(options)';

if isstruct(varargin{1})                    % Input as NEWOPTS struct
    newopts = varargin{1};
    for f = fields(newopts)'
        idx = find(strcmpi(f{:}, deff),1);
        if isempty(idx)
            error('parseOptions:unknownProperty', ...
                ['Unknown property ''' f{:} ''' in NEWOPTS.']);
        else
            options.(deff{idx}) = newopts.(f{:});
        end
    end
    varargin(1) = [];
end

if ~isempty(varargin)
    if ischar(varargin{1})                      % Input in name/value format
        % check for correct number of inputs
        if mod(numel(varargin),2) == 1
            error('parseOptions:wrongInputFormat', ...
                'Name and value input arguments must come in pairs.');
        end

        % parse arguments
        for i = 1:2:numel(varargin)
            if ischar(varargin{i})
                idx = find(strcmpi(varargin{i}, deff),1);
                if isempty(idx)
                    error('parseOptions:unknownProperty', ...
                        ['Unknown property name ''' varargin{i} '''.']);
                else
                    options.(deff{idx}) = varargin{i+1};
                end            
            else
                error('parseOptions:wrongInputFormat', ...
                    'Name and value input arguments must come in pairs.');
            end
        end
    else
        error('parseOptions:wrongInputFormat', ...
                'Input should come as a NEWOPTS struct and/or with name/value pairs.');
    end
end

end


