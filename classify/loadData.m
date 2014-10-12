function [feaVecs, labels] = loadData(path, featype)

if strcmp(featype, 'SURF')
    veclen = 800;
    strcmd = ['find ', path, ' -name "*800*.mat" > fea_WRAPUP.txt'];
elseif strcmp(featype, 'DECAF')
    veclen = 4096;
    strcmd = ['find ', path, ' -name "*.mat" > fea_WRAPUP.txt'];
else
    disp('wrong feature type, options: SURF or DECAF');
    feaVecs = [];
    labels = [];
    return;
end
system(strcmd);

% load features
fid = fopen('fea_WRAPUP.txt');
feaFiles = textscan(fid, '%s');

feaVecs = zeros(length(feaFiles{1}), veclen);
for i = 1:length(feaFiles{1})
    load(feaFiles{1}{i});
    if strcmp(featype, 'SURF')
        feaVecs(i,:) = histogram';
    elseif strcmp(featype, 'DECAF')
        feaVecs(i,:) = fc7';
    end
end
fclose(fid);

% normalization Very important !!!
norm2 = sqrt(sum(feaVecs.^2, 2) + 1e-6);
feaVecs = feaVecs./ repmat(norm2, 1, size(feaVecs, 2));

% construct labels
fid = fopen('fea_WRAPUP.txt');
feaFiles = textscan(fid, '%s%s%s%s%s%s', 'delimiter', '/');
labels = zeros(length(feaFiles{1}), 1);

uniqueCate = unique(feaFiles{5});
for i = 1:length(feaFiles{1})
    labels(i) = find(strcmp(uniqueCate, feaFiles{5}(i))) - 1;
end
fclose(fid);


%ind = find(strcmp(feaFiles{5},'back_pack') + ...
%           strcmp(feaFiles{5}, 'bike') + ...
%           strcmp(feaFiles{5}, 'calculator') + ...
%           strcmp(feaFiles{5}, 'headphones') + ...
%           strcmp(feaFiles{5}, 'keyboard') + ...
%           strcmp(feaFiles{5}, 'laptop_computer') + ...
%           strcmp(feaFiles{5}, 'monitor') + ...
%           strcmp(feaFiles{5}, 'mouse') + ...
%           strcmp(feaFiles{5}, 'mug') + ...
%           strcmp(feaFiles{5}, 'projector'));

%feaVecs = feaVecs(ind, :);
%labels = labels(ind, :);


end