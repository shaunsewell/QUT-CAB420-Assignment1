function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  
  if (d~=2) 
      error('Sorry -- plot2DLogistic only works on 2D data...'); 
  end

% Create the figure to plot the data on
figure('Name','2D Linear Classifier Plot');
hold on;

classes_in_Y = unique(Y)';
for c = classes_in_Y
    indicies_of_class_data = find(Y==c);
    x_values = X(indicies_of_class_data, 1:end);
    scatter(x_values(:, 1), x_values(:, 2), 'filled');  
end

% Plot decision boundary.
wts = getWeights(obj);
f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2;
fimplicit(f)
legend(strcat("Class ", num2str(classes_in_Y(1))), strcat("Class ", num2str(classes_in_Y(2))), 'Decision Boundary', 'FontSize',12);
hold off; 
