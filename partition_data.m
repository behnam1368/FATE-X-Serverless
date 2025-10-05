function nodeData = partition_data(X,Y,numNodes)
numSamples = size(X,1);
samplesPerNode = floor(numSamples / numNodes);
nodeData = cell(numNodes,1);
for i = 1:numNodes
    idxStart = (i-1)*samplesPerNode + 1;
    if i ~= numNodes
        idxEnd = i*samplesPerNode;
    else
        idxEnd = numSamples;
    end
    nodeData{i}.X = X(idxStart:idxEnd,:);
    nodeData{i}.Y = Y(idxStart:idxEnd,:);
end
end
