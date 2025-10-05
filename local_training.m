function [model, SHAP_vals] = local_training(X,Y,globalModel,lr,epochs,lambdaKD)
[numSamples,numFeatures] = size(X);
model.W1 = globalModel.W1;
model.b1 = globalModel.b1;
model.W2 = globalModel.W2;
model.b2 = globalModel.b2;

hiddenSize = size(model.W1,2);

Y = Y(:);

SHAP_vals = zeros(numSamples,1); % placeholder

for e = 1:epochs
    % Forward
    Z1 = X*model.W1 + repmat(model.b1,numSamples,1);
    A1 = max(0,Z1); % ReLU
    Z2 = A1*model.W2 + model.b2;
    probs = 1./(1+exp(-Z2)); % Sigmoid
    
    % Backward
    gradZ2 = probs - Y;
    gradW2 = (A1'*gradZ2)/numSamples;
    gradb2 = sum(gradZ2,1)/numSamples;
    
    gradA1 = gradZ2*model.W2';
    gradZ1 = gradA1 .* (Z1>0);
    gradW1 = (X'*gradZ1)/numSamples;
    gradb1 = sum(gradZ1,1)/numSamples;
    
    % Update
    model.W1 = model.W1 - lr*gradW1;
    model.b1 = model.b1 - lr*gradb1;
    model.W2 = model.W2 - lr*gradW2;
    model.b2 = model.b2 - lr*gradb2;
end
end
