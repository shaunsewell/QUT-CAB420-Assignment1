function mini_batches = create_mini_batches(obj, X,y, batch_size)

pi = randperm(size(X,1));
data_values = [X(pi,:), y(pi,:)];    %shuffle your data

n_mini_batches = size(X,1) / batch_size; %based on your data and the batch size compute the number of batches
mini_batches = zeros(batch_size,3,n_mini_batches);

for i = 1:n_mini_batches
   %extract the minibatch values
   start = batch_size * (i - 1) + 1;
   end_i = start + batch_size - 1;
   mini_batches(1:batch_size, :, i) = data_values(start:end_i,:);
end

end