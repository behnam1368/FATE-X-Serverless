function probs = softmax(scores)
    %SOFTMAX Compute softmax probabilities for MATLAB 2015a
    % scores: N x C (N samples, C classes) or 1xC (single sample)

    % Ensure scores is 2D
    if isvector(scores)
        scores = scores(:)'; % convert to 1 x C row vector
    end

    % Subtract max for numerical stability
    scores = scores - repmat(max(scores,[],2), 1, size(scores,2));

    % Exponentiate
    expScores = exp(scores);

    % Normalize
    rowSums = sum(expScores,2);
    probs = expScores ./ repmat(rowSums,1,size(expScores,2));
end
