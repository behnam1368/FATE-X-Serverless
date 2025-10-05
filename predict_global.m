function Y_pred = predict_global(globalModel,X)
Z1 = X*globalModel.W1 + repmat(globalModel.b1,size(X,1),1);
A1 = max(0,Z1);
Z2 = A1*globalModel.W2 + globalModel.b2;
probs = 1./(1+exp(-Z2));
Y_pred = probs>0.5;
end
