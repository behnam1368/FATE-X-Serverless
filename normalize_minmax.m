function Xnorm = normalize_minmax(X)
minVals = min(X,[],1);
maxVals = max(X,[],1);
denom = max(maxVals - minVals, eps);
Xnorm = (X - repmat(minVals,size(X,1),1)) ./ repmat(denom,size(X,1),1);
end
