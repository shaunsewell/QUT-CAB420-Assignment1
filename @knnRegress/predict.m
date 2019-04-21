% Test function: predict on Xtest
function Yte = predict(obj,Xte)
    % get size of training, test data
    [Ntr,Mtr] = size(obj.Xtrain);          
    [Nte,Mte] = size(Xte);
    
    % figure out how many classes & their labels
    classes = unique(obj.Ytrain);     
    
    % make Ytest the same data type as Ytrain
    Yte = repmat(obj.Ytrain(1), [Nte,1]); 
    
    K = min(obj.K, Ntr);  % can't have more than Ntrain neighbors
    
    for i=1:Nte                        
        % compute sum of squared differences
        dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);
        
        % find nearest neighbors over Xtrain (dimension 2)
        [tmp,idx] = sort(dist); 
        
        % idx(1) is the index of the nearest point, etc.

        % predict ith test example's value from nearest neighbors
        Yte(i)=mean(obj.Ytrain(idx(1:K)));       
    end
end