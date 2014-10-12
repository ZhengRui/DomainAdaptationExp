clear;clc;

% load surf features
% amazon
[Surf_amazon_fea_pool, Surf_amazon_lab_pool] = loadData('../SurfFeatures/amazon/', 'SURF');
% dslr
[Surf_dslr_fea_pool, Surf_dslr_lab_pool] = loadData('../SurfFeatures/dslr/', 'SURF');
% webcam
[Surf_webcam_fea_pool, Surf_webcam_lab_pool] = loadData('../SurfFeatures/webcam/', 'SURF');

% load decaf features
% amazon
[Decaf_amazon_fea_pool, Decaf_amazon_lab_pool] = loadData('../DecafFeatures/amazon/', 'DECAF');
% dslr
[Decaf_dslr_fea_pool, Decaf_dslr_lab_pool] = loadData('../DecafFeatures/dslr/', 'DECAF');
% webcam
[Decaf_webcam_fea_pool, Decaf_webcam_lab_pool] = loadData('../DecafFeatures/webcam/', 'DECAF');


precision = zeros(2, 5, 10);
for i = 1:10
    
    fprintf('****  iteration: %g ****\n', i);
    % split surf features
    % amazon
    [surf_amazon_train_fea, surf_amazon_train_lab, ~, ~] = split(Surf_amazon_fea_pool, Surf_amazon_lab_pool, 20);
    % dslr
    [surf_dslr_train_fea, surf_dslr_train_lab, ~, ~] = split(Surf_dslr_fea_pool, Surf_dslr_lab_pool, 8);
    % webcam
    [surf_web_train_fea, surf_web_train_lab, surf_web_test_fea, surf_web_test_lab] = split(Surf_webcam_fea_pool, Surf_webcam_lab_pool, 8);
    
    surf_a2w_model = svmtrain(surf_amazon_train_lab, surf_amazon_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(surf_web_test_lab, surf_web_test_fea, surf_a2w_model);
    precision(1,1,i) = accuracy(1);
    surf_d2w_model = svmtrain(surf_dslr_train_lab, surf_dslr_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(surf_web_test_lab, surf_web_test_fea, surf_d2w_model);
    precision(1,2,i) = accuracy(1);
    surf_w2w_model = svmtrain(surf_web_train_lab, surf_web_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(surf_web_test_lab, surf_web_test_fea, surf_w2w_model);
    precision(1,3,i) = accuracy(1);
    surf_aw2w_model = svmtrain([surf_amazon_train_lab; surf_web_train_lab], [surf_amazon_train_fea; surf_web_train_fea], '-s 1 -q');
    [~, accuracy, ~] = svmpredict(surf_web_test_lab, surf_web_test_fea, surf_aw2w_model);
    precision(1,4,i) = accuracy(1);
    surf_dw2w_model = svmtrain([surf_dslr_train_lab; surf_web_train_lab], [surf_dslr_train_fea; surf_web_train_fea], '-s 1 -q');
    [~, accuracy, ~] = svmpredict(surf_web_test_lab, surf_web_test_fea, surf_dw2w_model);
    precision(1,5,i) = accuracy(1);
   
    
    % split decaf features
    % amazon
    [decaf_amazon_train_fea, decaf_amazon_train_lab, ~, ~] = split(Decaf_amazon_fea_pool, Decaf_amazon_lab_pool, 20);
    % dslr
    [decaf_dslr_train_fea, decaf_dslr_train_lab, ~, ~] = split(Decaf_dslr_fea_pool, Decaf_dslr_lab_pool, 8);
    % webcam
    [decaf_web_train_fea, decaf_web_train_lab, decaf_web_test_fea, decaf_web_test_lab] = split(Decaf_webcam_fea_pool, Decaf_webcam_lab_pool, 8);
    
    decaf_a2w_model = svmtrain(decaf_amazon_train_lab, decaf_amazon_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(decaf_web_test_lab, decaf_web_test_fea, decaf_a2w_model);
    precision(2,1,i) = accuracy(1);
    decaf_d2w_model = svmtrain(decaf_dslr_train_lab, decaf_dslr_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(decaf_web_test_lab, decaf_web_test_fea, decaf_d2w_model);
    precision(2,2,i) = accuracy(1);
    decaf_w2w_model = svmtrain(decaf_web_train_lab, decaf_web_train_fea, '-s 1 -q');
    [~, accuracy, ~] = svmpredict(decaf_web_test_lab, decaf_web_test_fea, decaf_w2w_model);
    precision(2,3,i) = accuracy(1);
    decaf_aw2w_model = svmtrain([decaf_amazon_train_lab; decaf_web_train_lab], [decaf_amazon_train_fea; decaf_web_train_fea], '-s 1 -q');
    [~, accuracy, ~] = svmpredict(decaf_web_test_lab, decaf_web_test_fea, decaf_aw2w_model);
    precision(2,4,i) = accuracy(1);
    decaf_dw2w_model = svmtrain([decaf_dslr_train_lab; decaf_web_train_lab], [decaf_dslr_train_fea; decaf_web_train_fea], '-s 1 -q');
    [~, accuracy, ~] = svmpredict(decaf_web_test_lab, decaf_web_test_fea, decaf_dw2w_model);
    precision(2,5,i) = accuracy(1);
    
end

disp('***** classification result *****');
fprintf('a -> w, surf: %f +- %f, decaf: %f +- %f\n', mean(precision(1,1,:)), std(precision(1,1,:)), mean(precision(2,1,:)), std(precision(2,1,:)));
fprintf('d -> w, surf: %f +- %f, decaf: %f +- %f\n', mean(precision(1,2,:)), std(precision(1,2,:)), mean(precision(2,2,:)), std(precision(2,2,:)));
fprintf('w -> w, surf: %f +- %f, decaf: %f +- %f\n', mean(precision(1,3,:)), std(precision(1,3,:)), mean(precision(2,3,:)), std(precision(2,3,:)));
fprintf('aw -> w, surf: %f +- %f, decaf: %f +- %f\n', mean(precision(1,4,:)), std(precision(1,4,:)), mean(precision(2,4,:)), std(precision(2,4,:)));
fprintf('dw -> w, surf: %f +- %f, decaf: %f +- %f\n', mean(precision(1,5,:)), std(precision(1,5,:)), mean(precision(2,5,:)), std(precision(2,5,:)));



