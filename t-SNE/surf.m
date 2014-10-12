clear, clc;
system('find "../SurfFeatures/" -name "*800*.mat" -not -path "../SurfFeatures/amazon/*" > fea_WRAPUP.txt');

% load feature vectors
fid = fopen('fea_WRAPUP.txt');
feaFiles = textscan(fid, '%s');

feaVecs = zeros(length(feaFiles{1}), 800);
for i = 1:length(feaFiles{1})
    load(feaFiles{1}{i});
    feaVecs(i,:) = histogram';
end

fclose(fid);


% construct labels
fid = fopen('fea_WRAPUP.txt');
feaFiles = textscan(fid, '%s%s%s%s%s%s', 'delimiter', '/');
labels = zeros(length(feaFiles{1}), 1);

uniqueDomain = unique(feaFiles{3});
uniqueCate = unique(feaFiles{5});

for i = 1:length(feaFiles{1})
    labels(i) = 2 * find(strcmp(uniqueCate, feaFiles{5}(i)))+ ...
                find(strcmp(uniqueDomain, feaFiles{3}(i))) - 3;   
end

% run tsne clustering
no_dims = 2;
init_dims = 30;
perplexity = 30;
mappedX = tsne(feaVecs, labels, no_dims, init_dims, perplexity);
%gscatter(mappedX(:,1), mappedX(:,2), labels);

% write to file for displaying images overlayed with each other
x_min = min(mappedX(:, 1));
x_max = max(mappedX(:, 1));
y_min = min(mappedX(:, 2));
y_max = max(mappedX(:, 2));
BIGIMG = ones(int16(x_max - x_min) * 10 + 60, ...
    int16(y_max - y_min) * 10 + 60, 3) * 256;

wid = fopen('tsne.res', 'w');
for i = 1:length(feaFiles{1})
    imgSrc = strcat('../thumbnails/30x30/',feaFiles{3}{i},'/images/',feaFiles{5}{i}, '/frame_', feaFiles{6}{i}(11:14),'.jpg');
    fprintf(wid, '%f %f %s\n', mappedX(i, 1), mappedX(i, 2), imgSrc);
    img = imread(imgSrc);
    x_start = int16(mappedX(i, 1) - x_min) * 10 + 15;
    y_start = int16(mappedX(i, 2) - y_min) * 10 + 15;
    BIGIMG(x_start:(x_start + 29), y_start:(y_start+29), :) = img;
end

fclose(fid);
fclose(wid);

BIGIMG = BIGIMG/256;
imshow(BIGIMG);
imwrite(BIGIMG, 'surftsne.jpg', 'jpg');