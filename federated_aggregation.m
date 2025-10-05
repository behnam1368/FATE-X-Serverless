function globalModel = federated_aggregation(localModels,nodeData)
numNodes = length(localModels);
totalSamples = sum(cellfun(@(c) size(c.X,1), nodeData));

% Weighted average
globalModel.W1 = zeros(size(localModels{1}.W1));
globalModel.b1 = zeros(size(localModels{1}.b1));
globalModel.W2 = zeros(size(localModels{1}.W2));
globalModel.b2 = 0;

for i = 1:numNodes
    ni = size(nodeData{i}.X,1);
    globalModel.W1 = globalModel.W1 + (ni/totalSamples)*localModels{i}.W1;
    globalModel.b1 = globalModel.b1 + (ni/totalSamples)*localModels{i}.b1;
    globalModel.W2 = globalModel.W2 + (ni/totalSamples)*localModels{i}.W2;
    globalModel.b2 = globalModel.b2 + (ni/totalSamples)*localModels{i}.b2;
end
end
