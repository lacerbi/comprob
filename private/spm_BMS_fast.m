function [alpha,exp_r,xp,pxp,bor,g] = spm_BMS_fast(lme, Nsamp, do_plot, sampling, ecp, alpha0)
% Bayesian model selection for group studies (vectorized for speed)
% FORMAT [alpha,exp_r,xp,pxp,bor] = spm_BMS_fast (lme, Nsamp, do_plot, sampling, ecp, alpha0)
% 
% INPUT:
% lme      - array of log model evidences 
%              rows: subjects
%              columns: models (1..Nk)
% Nsamp    - number of samples used to compute exceedance probabilities
%            (default: 1e6)
% do_plot  - 1 to plot p(r|y)
% sampling - use sampling to compute exact alpha
% ecp      - 1 to compute exceedance probability
% alpha0   - [1 x Nk] vector of prior model counts
% 
% OUTPUT:
% alpha   - vector of model probabilities
% exp_r   - expectation of the posterior p(r|y)
% xp      - exceedance probabilities
% pxp     - protected exceedance probabilities
% bor     - Bayes Omnibus Risk (probability that model frequencies 
%           are equal)
% g       - matrix of individual posterior probabilities
% 
% REFERENCES:
%
% Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
% Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
%
% Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
% Bayesian model selection for group studiesóRevisited. 
% NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% 2016 Modified by Luigi Acerbi for speed and added interface

% Klaas Enno Stephan, Will Penny, Lionel Rigoux and J. Daunizeau
% $Id: spm_BMS.m 5842 2014-01-20 10:53:17Z will $

if nargin < 2 || isempty(Nsamp)
    Nsamp = 1e6;
end
if nargin < 3 || isempty(do_plot)
    do_plot = 0;
end
if nargin < 4 || isempty(sampling)
    sampling = 0;
end
if nargin < 5 || isempty(ecp)
    ecp = (nargout > 2);
end

max_val = log(realmax('double'));
Ni      = size(lme,1);  % number of subjects
Nk      = size(lme,2);  % number of models
c       = 1;
cc      = 10e-6;


% prior observations
%--------------------------------------------------------------------------
if nargin < 6 || isempty(alpha0)
    alpha0  = ones(1,Nk);    
end
alpha   = alpha0;


% iterative VB estimation
%--------------------------------------------------------------------------
while c > cc,

    % compute posterior belief g(i,k)=q(m_i=k|y_i) that model k generated
    % the data for the i-th subject

    % integrate out prior probabilities of models (in log space)
    log_u = bsxfun(@plus, lme, psi(alpha)- psi(sum(alpha)));    
    log_u = bsxfun(@minus, log_u, mean(log_u,2));

    % prevent numerical problems for badly scaled posteriors
    log_u = sign(log_u) .* min(max_val,abs(log_u));
   
    % exponentiate (to get back to non-log representation)
    u  = exp(log_u);

    % normalisation: sum across all models for i-th subject
    u_i     = sum(u,2);
    g       = bsxfun(@rdivide, u, u_i);
                
    % expected number of subjects whose data we believe to have been 
    % generated by model k
    beta = sum(g,1);

    % update alpha
    prev  = alpha;
    alpha = alpha0 + beta;
    
    % convergence?
    c = norm(alpha - prev);

end


% Compute expectation of the posterior p(r|y)
%--------------------------------------------------------------------------
exp_r = alpha./sum(alpha);


% Compute exceedance probabilities p(r_i>r_j), Bayesian Omnibus Risk, and
% protected exceedance probabilities
%--------------------------------------------------------------------------
if ecp
    [xp,bor,pxp] = spm_dirichlet_exceedance_fast(alpha, Nsamp, lme, g, alpha0);
else
    xp = []; bor = []; pxp = [];
end


% Graphics output (currently for 2 models only)
%--------------------------------------------------------------------------
if do_plot && Nk == 2
    % plot Dirichlet pdf
    %----------------------------------------------------------------------
    if alpha(1)<=alpha(2)
       alpha_now =sort(alpha,1,'descend');
       winner_inx=2;
    else
        alpha_now =alpha;
       winner_inx=1;
    end
    
    x1  = [0:0.0001:1];
    for i = 1:length(x1),
        p(i)   = spm_Dpdf([x1(i) 1-x1(i)],alpha_now);
    end
    fig1 = figure;
    axes1 = axes('Parent',fig1,'FontSize',14);
    plot(x1,p,'k','LineWidth',1);
    % cumulative probability: p(r1>r2)
    i  = find(x1 >= 0.5);
    hold on
    fill([x1(i) fliplr(x1(i))],[i*0 fliplr(p(i))],[1 1 1]*.8)
    v = axis;
    plot([0.5 0.5],[v(3) v(4)],'k--','LineWidth',1.5);
    xlim([0 1.05]);
    xlabel(sprintf('r_%d',winner_inx),'FontSize',18);
    ylabel(sprintf('p(r_%d|y)',winner_inx),'FontSize',18);
    title(sprintf('p(r_%d>%1.1f | y) = %1.3f',winner_inx,0.5,xp(winner_inx)),'FontSize',18);
    legend off
end


% Sampling approach ((currently implemented for 2 models only):
% plot F as a function of alpha_1
%--------------------------------------------------------------------------
if sampling
    if Nk == 2
        % Compute lower bound on F by sampling
        %------------------------------------------------------------------
        alpha_max = size(lme,1) + Nk*alpha0(1);
        dx        = 0.1;
        a         = [1:dx:alpha_max];
        Na        = length(a);
        for i=1:Na,
            alpha_s                = [a(i),alpha_max-a(i)];
            [F_samp(i),F_bound(i)] = spm_BMS_F(alpha_s,lme,alpha0);
        end
        if do_plot
        % graphical display
        %------------------------------------------------------------------
        fig2 = figure;
        axes2 = axes('Parent',fig2,'FontSize',14);
        plot(a,F_samp,'Parent',axes2,'LineStyle','-','DisplayName','Sampling Approach',...
            'Color',[0 0 0]);
        hold on;
        yy = ylim;
        plot([alpha(1),alpha(1)],[yy(1),yy(2)],'Parent',axes2,'LineStyle','--',...
            'DisplayName','Variational Bayes','Color',[0 0 0]);
        legend2 = legend(axes2,'show');
        set(legend2,'Position',[0.15 0.8 0.2 0.1],'FontSize',14);
        xlabel('\alpha_1','FontSize',18);
        ylabel('F','FontSize',18);
        end
    else
        fprintf('\n%s\n','Verification of alpha estimates by sampling not available.')
        fprintf('%s\n','This approach is currently only implemented for comparison of 2 models.');
    end
end

function [xp,bor,pxp] = spm_dirichlet_exceedance_fast(alpha, Nsamp, lme, g, alpha0)
% Compute exceedance probabilities and related quantities for a Dirichlet distribution
% FORMAT xp = spm_dirichlet_exceedance_fast(alpha,Nsamp)
% 
% Input:
% alpha     - Dirichlet parameters
% Nsamp     - number of samples used to compute xp [default = 1e6]
% lme      - array of log model evidences 
%              rows: subjects
%              columns: models (1..Nk)
% g       - matrix of individual posterior probabilities
% alpha0   - [1 x Nk] vector of prior model counts
% 
% Output:
% xp        - exceedance probability
% bor       - Bayes Omnibus Risk (probability that model frequencies 
%             are equal)
% pxp       - protected exceedance probabilities
%__________________________________________________________________________
%
% This function computes exceedance probabilities, i.e. for any given model
% k1, the probability that it is more likely than any other model k2.  
% More formally, for k1=1..Nk and for all k2~=k1, it returns p(x_k1>x_k2) 
% given that p(x)=dirichlet(alpha).
% If requested (second and third outputs), it also computes the Bayes
% Omnibus Risk (probability that model frequencies are equal), and protected
% exceedance probabilities (probability that any one model is more 
% frequent than the others, above and beyond chance).
% 
% Refs:
% Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ
% Bayesian Model Selection for Group Studies. NeuroImage (2008)
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Based on code by Will Penny & Klaas Enno Stephan
% Modified by Luigi Acerbi 2016
% $Id: spm_dirichlet_exceedance.m 3118 2009-05-12 17:37:32Z guillaume $

if nargin < 2
    Nsamp = 1e6;
end

Nk = length(alpha);

% Compute exceedance probabilities p(r_i>r_j)
%--------------------------------------------------------------------------
if Nk == 2
    % comparison of 2 models
    xp(1) = spm_Bcdf_private(0.5,alpha(2),alpha(1));
    xp(2) = spm_Bcdf_private(0.5,alpha(1),alpha(2));
else
    % comparison of >2 models: use sampling approach

    % Perform sampling in blocks
    %--------------------------------------------------------------------------
    blk = ceil(Nsamp*Nk*8 / 2^28);
    blk = floor(Nsamp/blk * ones(1,blk));
    blk(end) = Nsamp - sum(blk(1:end-1));

    xp = zeros(1,Nk);
    for i=1:length(blk)

        % Sample from univariate gamma densities then normalise
        % (see Dirichlet entry in Wikipedia or Ferguson (1973) Ann. Stat. 1,
        % 209-230)
        %----------------------------------------------------------------------

        r = gamrnd(repmat(alpha,[blk(i),1]),1,blk(i),Nk);    
        r = bsxfun(@rdivide, r, sum(r,2));

        % Exceedance probabilities:
        % For any given model k1, compute the probability that it is more
        % likely than any other model k2~=k1
        %----------------------------------------------------------------------
        [y, idx] = max(r,[],2);
        xp = xp + histc(idx, 1:Nk)';

    end
    xp = xp / Nsamp;
end

% Compute Bayes Omnibus Risk - use functions from VBA toolbox
if nargout > 1
    posterior.a=alpha;
    posterior.r=g';
    priors.a=alpha0;

    F1 = FE(lme',posterior,priors); % Evidence of alternative
    
    options.families=[];
    F0 = FE_null(lme',options); % Evidence of null (equal model freqs)
    
    % Implied by Eq 5 (see also p39) in Rigoux et al.
    % See also, last equation in Appendix 2
    bor=1/(1+exp(F1-F0));

    % Compute protected exceedance probs - Eq 7 in Rigoux et al.
    if nargout > 2
        pxp=(1-bor)*xp+bor/Nk;
    end
end


function F = spm_Bcdf_private(x,v,w)
% Inverse Cumulative Distribution Function (CDF) of Beta distribution
% FORMAT F = spm_Bcdf(x,v,w)
%
% x   - Beta variates (Beta has range [0,1])
% v   - Shape parameter (v>0)
% w   - Shape parameter (w>0)
% F   - CDF of Beta distribution with shape parameters [v,w] at points x
%__________________________________________________________________________
%
% spm_Bcdf implements the Cumulative Distribution Function for Beta
% distributions.
%
% Definition:
%--------------------------------------------------------------------------
% The Beta distribution has two shape parameters, v and w, and is
% defined for v>0 & w>0 and for x in [0,1] (See Evans et al., Ch5).
% The Cumulative Distribution Function (CDF) F(x) is the probability
% that a realisation of a Beta random variable X has value less than
% x. F(x)=Pr{X<x}: This function is usually known as the incomplete Beta
% function. See Abramowitz & Stegun, 26.5; Press et al., Sec6.4 for
% definitions of the incomplete beta function.
%
% Variate relationships:
%--------------------------------------------------------------------------
% Many: See Evans et al., Ch5
%
% Algorithm:
%--------------------------------------------------------------------------
% Using MATLAB's implementation of the incomplete beta finction (betainc).
%
% References:
%--------------------------------------------------------------------------
% Evans M, Hastings N, Peacock B (1993)
%       "Statistical Distributions"
%        2nd Ed. Wiley, New York
%
% Abramowitz M, Stegun IA, (1964)
%       "Handbook of Mathematical Functions"
%        US Government Printing Office
%
% Press WH, Teukolsky SA, Vetterling AT, Flannery BP (1992)
%       "Numerical Recipes in C"
%        Cambridge
%__________________________________________________________________________
% Copyright (C) 1999-2011 Wellcome Trust Centre for Neuroimaging

% Andrew Holmes
% $Id: spm_Bcdf.m 4182 2011-02-01 12:29:09Z guillaume $


%-Format arguments, note & check sizes
%--------------------------------------------------------------------------
if nargin<3, error('Insufficient arguments'), end

ad = [ndims(x);ndims(v);ndims(w)];
rd = max(ad);
as = [[size(x),ones(1,rd-ad(1))];...
      [size(v),ones(1,rd-ad(2))];...
      [size(w),ones(1,rd-ad(3))]];
rs = max(as);
xa = prod(as,2)>1;
if sum(xa)>1 && any(any(diff(as(xa,:)),1))
    error('non-scalar args must match in size');
end

%-Computation
%--------------------------------------------------------------------------
%-Initialise result to zeros
F = zeros(rs);

%-Only defined for x in [0,1] & strictly positive v & w.
% Return NaN if undefined.
md = ( x>=0  &  x<=1  &  v>0  &  w>0 );
if any(~md(:))
    F(~md) = NaN;
    warning('Returning NaN for out of range arguments');
end

%-Special cases: F=1 when x=1
F(md & x==1) = 1;

%-Non-zero where defined & x>0, avoid special cases
Q  = find( md  &  x>0  &  x<1 );
if isempty(Q), return, end
if xa(1), Qx=Q; else Qx=1; end
if xa(2), Qv=Q; else Qv=1; end
if xa(3), Qw=Q; else Qw=1; end

%-Compute
F(Q) = betainc(x(Qx),v(Qv),w(Qw));


function [F,ELJ,Sqf,Sqm] = FE(L,posterior,priors)
% derives the free energy for the current approximate posterior
% This routine has been copied from the VBA_groupBMC function
% of the VBA toolbox http://code.google.com/p/mbb-vb-toolbox/ 
% and was written by Lionel Rigoux and J. Daunizeau
%
% See equation A.20 in Rigoux et al. (should be F1 on LHS)

[K,n] = size(L);
a0 = sum(posterior.a);
Elogr = psi(posterior.a) - psi(sum(posterior.a));
Sqf = sum(gammaln(posterior.a)) - gammaln(a0) - sum((posterior.a-1).*Elogr);
Sqm = 0;
for i=1:n
    Sqm = Sqm - sum(posterior.r(:,i).*log(posterior.r(:,i)+eps));
end
ELJ = gammaln(sum(priors.a)) - sum(gammaln(priors.a)) + sum((priors.a-1).*Elogr);
for i=1:n
    for k=1:K
        ELJ = ELJ + posterior.r(k,i).*(Elogr(k)+L(k,i));
    end
end
F = ELJ + Sqf + Sqm;



function [F0m,F0f] = FE_null(L,options)
% derives the free energy of the 'null' (H0: equal model frequencies)
% This routine has been copied from the VBA_groupBMC function
% of the VBA toolbox http://code.google.com/p/mbb-vb-toolbox/ 
% and was written by Lionel Rigoux and J. Daunizeau
%
% See Equation A.17 in Rigoux et al.

[K,n] = size(L);
if ~isempty(options.families)
    f0 = options.C*sum(options.C,1)'.^-1/size(options.C,2);
    F0f = 0;
else
    F0f = [];
end
F0m = 0;
for i=1:n
    tmp = L(:,i) - max(L(:,i));
    g = exp(tmp)./sum(exp(tmp));
    for k=1:K
        F0m = F0m + g(k).*(L(k,i)-log(K)-log(g(k)+eps));
        if ~isempty(options.families)
            F0f = F0f + g(k).*(L(k,i)-log(g(k))+log(f0(k)));
        end
    end
end






