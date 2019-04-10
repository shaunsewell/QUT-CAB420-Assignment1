function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

wts = getWeights(obj);

f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2;

function_data = sign(f(Xte(:,1), Xte(:,2)));

Yte = obj.classes(ceil((function_data+3)/2));
end