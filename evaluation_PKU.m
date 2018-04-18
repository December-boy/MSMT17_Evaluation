clc;clear all;close all;

addpath(genpath('utils/'));
addpath(genpath('LOMO_XQDA/'));
%run('KISSME/toolbox/init.m');

list_query = importdata('list_query.txt');
list_query = list_query.textdata();
list_gallery = importdata('list_gallery.txt');
list_gallery = list_gallery.textdata();



%gt_dir = 'dataset\gt_bbox\'; % directory of hand-drawn bounding boxes
addpath 'CM_curve/' % draw confusion matrix
gallery_ID = importdata('data/gallery_ID.mat');
gallery_CAM = importdata('data/gallery_CAM.mat');    

query_ID = importdata('data/query_ID.mat'); 
query_CAM = importdata('data/query_CAM.mat');  

query_feature = load('your_query_feature');
query_feature = query_feature';
gallery_feature = load('your_gallyer_feature');
gallery_feature = gallery_feature';


nQuery = size(query_feature,2);
nTest = size(gallery_feature,2);



%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision

CMC = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query

dist = sqdist(gallery_feature, query_feature); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized


knn = 1; % number of expanded queries. knn = 1 yields best result

for k = 1:nQuery
    k
    % load groud truth for each query (good and junk)
    good_index = intersect(find(gallery_ID == query_ID(k)), find(gallery_CAM ~= query_CAM(k)))';% images with the same ID but different camera from the query
    
    junk_index = intersect(find(gallery_ID == query_ID(k)), find(gallery_CAM == query_CAM(k))); % images with the same ID and the same camera as the query
    tic
    score = dist(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    
    
end

CMC = mean(CMC);
%% print result
fprintf('single query:                                   mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));
figure;
s = 50;
CMC_curve = [CMC ];
plot(1:s, CMC_curve(:, 1:s));





