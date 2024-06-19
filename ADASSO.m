function [B, C, D, score] = ADASSO(A, F, K, tau, err_ov)
% [B, C, D, score] = ADASSO(A, F, K, tau, err_ov)
%
% Asymmetric Discriminative-Associative (ADASSO) algorithm.
% A      = (L * M) foreground input Boolean matrix in unipolar form {1,0}.
% F      = (L * N) background input Boolean matrix in unipolar form {1,0}.
% K      = number of bases.
% tau    = discriminative association confidence threshold.
% err_ov = tolerable overcoverage error ratio (default err_ov = 0.5).
% B     = (L * K) basis Boolean matrix in unipolar form {1,0}.
% C     = (K * M) occurrence Boolean matrix corresponding to A in unipolar form {1,0}.
% D     = (K * N) occurrence Boolean matrix corresponding to F in unipolar form {1,0}.
% score = (K * 1) optimization scores.

[L, M] = size(A);
N      = size(F,2);
B      = zeros(L,K);
C      = zeros(K,M);
score  = zeros(K,1);
if nargout > 2
    D      = zeros(K,N);
    cond_D = true;
else
    cond_D = false;
end
if nargin < 5
    w_ov = 1;
else
    w_ov = 1/err_ov - 1;
end
wA0                         = w_ov * double(~A);
a                           = max(sum(A,2)',1);
f                           = max(sum(F,2)',1);
H                           = double(((A*A')./a-(F*F')./f) >= tau);
Hn1                         = sum(H,1)';
R                           = H' * F;
R((R-w_ov*H'*double(~F))<0) = 0;
Rsum                        = sum(R,2) / N;
INDS                        = 1:L;
for k = 1:K
    Q   = max(H'*(A-wA0), 0);
    arg = sum(Q,2)/M - Rsum;
    ind = arg == max(arg);
    if sum(ind) > 1
        ind = and(Hn1==min(Hn1(ind)), ind);
        j   = min(INDS(ind));
    else
        j = INDS(ind);
    end
    B(:,k)   = H(:,j);
    C(k,:)   = double(Q(j,:) > 0);
    score(k) = arg(j);
    if cond_D
        D(k,:) = double(R(j,:) > 0);
    end
    A = double(and(A, ~(B(:,k)*C(k,:))));
end
return