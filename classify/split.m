function [train_fea, train_lab, test_fea, test_lab] = split(feaVecs, labels, N_Per_Class)

labelUnique = unique(labels);

train_num = 0;
for i = 1:length(labelUnique)
    labelInd = find(labels == labelUnique(i));
    if length(labelInd) < N_Per_Class
        train_num = train_num + length(labelInd);
    else
        train_num = train_num + N_Per_Class;
    end
end

train_ind = zeros(train_num, 1);
test_ind = zeros(length(labels) - train_num, 1);

train_i = 1;
test_i = 1;
for i = 1:length(labelUnique)
    labelInd = find(labels == labelUnique(i));
    randInd = randperm(length(labelInd));
    train_incre = min(N_Per_Class, length(labelInd));
    test_incre = length(labelInd) - train_incre;
    train_ind(train_i:train_i+train_incre-1)  = labelInd(randInd(1:train_incre));
    if test_incre >= 1
        test_ind(test_i:test_i+test_incre-1) = labelInd(randInd(train_incre+1:end));
    end
    train_i = train_i + train_incre;
    test_i = test_i + test_incre;
end

train_fea = feaVecs(train_ind,:);
train_lab = labels(train_ind);
test_fea = feaVecs(test_ind, :);
test_lab = labels(test_ind);
%disp([train_ind(1:10), test_ind(1:10)]);
end