function nodeData = split_edge_nodes(X, Y, numNodes)
    %SPLIT_EDGE_NODES Split dataset among edge nodes randomly

    numSamples = size(X,1);
    indices = randperm(numSamples);

    % ������ ����� ������� ���? �� ���
    samplesPerNode = floor(numSamples / numNodes);
    nodeData = cell(numNodes,1);

    for i = 1:numNodes
        startIdx = (i-1)*samplesPerNode + 1;
        if i ~= numNodes
            endIdx = i*samplesPerNode;
        else
            endIdx = numSamples; % ���?� ��� ��� ���?������
        end
        idx = indices(startIdx:endIdx);
        nodeData{i}.X = X(idx,:);
        nodeData{i}.Y = Y(idx);
    end
end
