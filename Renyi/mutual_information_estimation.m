function mutual_information = mutual_information_estimation(variable1,variable2,sigma1,sigma2,alpha)

%% variable 1 is class labels
%  variable 2 is feature set (contains one or more features)

%% estimate entropy for variable 1, i.e., y
K_y = real(guassianMatrix(variable1,sigma1))/size(variable1,1);
[~, L_y] = eig(K_y);
lambda_y = abs(diag(L_y));
H_y = (1/(1-alpha))*log((sum(lambda_y.^alpha)));    

%% estimate entropy for variable 2, i.e., x = [s1, s2, s3, ...]
% estimate entropy for each feature, i.e., H(s1), H(s2), H(s3), ...
L = size(variable2,2);
K_s = [];
H_s = zeros(1,L);
for i=1:L
    source = variable2(:,i);
    K_s{i} = real(guassianMatrix(source,sigma2))/size(source,1);
    [~, L_s] = eig(K_s{i});
    lambda_s = abs(diag(L_s));
    H_s(i) = (1/(1-alpha))*log((sum(lambda_s.^alpha)));
 end   

% estimate entropy H(x) = H(s1,s2,s3...) by Hadamard product
K_sall = K_s{1};
for i=2:L
    K_sall = K_sall.*K_s{i}*size(variable1,1);
end
[~,L_sall] = eig(K_sall);
lambda_sall = abs(diag(L_sall));
H_sall =  (1/(1-alpha))*log( (sum(lambda_sall.^alpha)));
    
%% estimate joint entropy H(y,s1,s2,s3...)
K_ysall = K_y.*K_sall.*size(variable1,1);
[~,L_ysall] = eig(K_ysall);
lambda_ysall = abs(diag(L_ysall));
H_ysall =  (1/(1-alpha))*log( (sum(lambda_ysall.^alpha)));
    
%% estimate mutual information estimation Y gained from S1, S2 ,,,
% I(y;{s1,s2,s3,...}) = H(y) + H(s1,s2,s3,...) - H(y,s1,s2,s3,...)
mutual_information = H_y + H_sall - H_ysall;

end