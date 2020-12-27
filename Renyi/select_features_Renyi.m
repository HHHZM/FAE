function [sel_flag] = select_features_Renyi(train_data,train_labels,number_features)

%% sel_flag (a vector) contains selected feature index
%% Acc_vec (a vector) records classification accuracy w.r.t different # of features

sel_flag = [];
sel_remain = [1:size(train_data,2)];
sel_remain = setdiff(sel_remain,sel_flag); % index of remaining features

% sigma = floor(sqrt(size(train_data,1)));
sigma1 = 1; % kernel size for labels, i.e., y 
sigma2 = 1; % kernel size for features, i.e., x1, x2, x3, ...
% in the T-PAMI paper, we set sigma1 = sigma2 = 1
% sigma2 can be tuned with Silverman's rule of thumb that considers the
% variance of x, or with 10 to 20 percent of the total (median) range of the Euclidean
% distances between all pairwise data points (pls refer details in the paper)
alpha = 1.01; % order of Renyi's entropy

% Encode categorical integer labels as a one-hot numeric array
extLabels = zeros(numel(unique(train_labels)),size(train_data,1));
extLabels(sub2ind(size(extLabels), train_labels', 1:size(train_data,1))) = 1;

for i=1:number_features
    %% select feature
    fprintf('Selecting the %d-th feature \n',i);
    MI_vector = zeros(1,size(train_data,2)-i+1); 
    % MI_vector records the mutual information values of different
    % candidate features (in conjunction with the selected features) and class labels.
    % e.g., suppose there are 5 features, x1 and x4 are selected,
    % MI_vector records I({x1,x2,x4};y), I({x1,x3,x4};y), and I({x1,x5,x4};y)
    for j=1:numel(MI_vector)
%         j
        variable1 = train_data(:,[sel_flag sel_remain(j)]);
        MI_vector(j) = mutual_information_estimation(extLabels',variable1,sigma1,sigma2,alpha);
    end
    index = find(MI_vector==max(MI_vector)); 
    % determine feature candidate that outputs the largest mutual information value
    index = index(1);  % select the first selected feature if 
                       % there are some features report same MI values
    sel_flag(end+1) = sel_remain(index);
    sel_remain(sel_remain == sel_flag(end))=[];
    clear MI_vector
    
end

end