%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

load('classifierdata.mat', "centroids", "centroidLabels");

test=csvread('mnist_test_200.csv');
correctlabels = test(:,785);

predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);
test(i,785) = prediction_index;

predictions(i) = centroidLabels(prediction_index);


end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0

% Creates a vector with the mean distances of each cluster to it's centroid
mean_Distance = zeros(size(centroids,1),1);
outlier_threshold = 1.3;

for i = 1:size(centroids,1)
    cluster = test(test(:, 785) == i, 1:784);
    distances = zeros(size(cluster,1), 1);
    for j = 1:size(cluster, 1)
        distances(j) = norm(cluster(j, :) - centroids(i, :));
    end
    mean_Distance(i) = mean(distances);
end

for i = 1:size(test, 1)
    distance_to_cetroid = norm(test(i,1:784) - centroids(test(i,785), :));
    if distance_to_cetroid > outlier_threshold * mean_Distance(test(i,785))
        outliers(i) = 1;
    end
end


disp("Outliers: " + sum(outliers));

%% Creates figure with all misclassified images
misclassified = test(predictions(:) ~= correctlabels(:),1:784);

figure;
colormap('gray');

plotsize = ceil(sqrt(size(misclassified(),1)));

for ind = 1:size(misclassified(),1)
    image = misclassified(ind,(1:784));
    subplot(plotsize,plotsize,ind);

    imagesc(reshape(image,[28 28])');
    title(strcat('Classified as ', num2str(predictions(ind))))
end



%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers);
title('Outlier Flags');
xlabel('Test Image');
ylabel('Outlier Flag');

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equal to the corresponding entry in prediction
% However, remember that some of these are outliers
disp("Accuracy: " + sum(correctlabels==predictions)/size(test,1));

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
    num_centroids = size(centroids, 1);
    distances = zeros(num_centroids: 1);

    for i = 1:num_centroids
        distances(i) = norm(data(1:784) - centroids(i, (1:784)));
    end

    [vec_distance, index] = min(distances);

end


