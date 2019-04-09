%% CAB420 Assignment 1 
%   Shaun Sewell N9509623 
%
%% 1. Features, Classes, and Linear Regression
clear ; close all; clc
% (a) Plot the training data in a scatter plot.
mTrain = load('data/mTrainData.txt'); % Load training data
Xtr = mTrain(: ,1); Ytr = mTrain(: ,2); % Separate features

figure('name', 'Scatter Plot of Training Data');
plot (Xtr, Ytr, 'bo'); % Plot training data
xlabel('X');
ylabel('Y');
title('Scatter Plot of Training Data');

% (b) Create a linear regression learner using the above functions. Plot it on
%the same plot as the training data.
Xtr_2 =  polyx(Xtr,2);
regress_learner = linearReg(Xtr_2 ,Ytr); % train a linear regression learner
xline = [0:.01:2]' ; % transpose : make a column vector , like training x
yline = predict( regress_learner , polyx (xline ,2) ); % assuming quadratic features

figure('name', 'Linear Regression predictor');
plot(xline, yline, 'ro');
hold on % Plot training data and label figure.
plot (Xtr, Ytr, 'bo');
xlabel('X');
ylabel('Y');
legend('Linear Predictor', 'Training Data');
title('Linear Regression predictor');

% (c) Create plots with the data and a higher-order polynomial (3, 5, 7, 9, 11, 13)
polys = [3, 5, 7, 9, 11, 13];

for p=1:size(polys)
    Xtr_p =  polyx(Xtr,polys(p));
    learner = linearReg(Xtr_p ,Ytr);                    % train a linear regression learner
    xline = [0:.01:2]' ;                                % transpose : make a column vector , like training x
    yline = predict( learner , polyx (xline ,polys(p)) ); % assuming quadratic features
    
    name = 'Predictor using higher order polynomial ' + polys(p);
    figure('name', name);
    plot(xline, yline, 'ro');
    hold on % Plot training data and label figure.
    plot (Xtr, Ytr, 'bo');
    xlabel('X');
    ylabel('Y');
    legend('Linear Predictor', 'Training Data');
    title('Linear Regression predictor');
end


