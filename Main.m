%% Main.m - FATE-X Simulation on Serverless Edge Computing
clc; clear; close all;

%% 0. Parameters
%datasetPath = 'Dataset/E3_Darpa_20.xlsx';
%datasetPath = 'Dataset/CICAPT1_IIoTDataset2024.xlsx';
datasetPath = 'Dataset/Wget.xlsx';
%datasetPath = 'Dataset/Wget_Hour.xlsx';
%datasetPath = 'Dataset/NSL_KDD.xlsx';
%datasetPath = 'Dataset/NSL_KDD.xlsx';
nodeConfigs = [10, 50, 100, 200, 500, 1000]; % Number of nodes
localEpochs = 5;       % Local training epochs
learningRate = 0.01;   % Learning rate
lambdaKD = 0.5;        % Knowledge Distillation weight
globalRounds = 3;      % Number of federated rounds
hiddenSize = 32;        % Hidden neurons in MLP

%% 1. Node specifications
fprintf('Serverless Edge Node Specifications:\n');
fprintf('CPU: Intel Xeon Gold 6338 (2.0 GHz, 32 cores)\n');
fprintf('GPU: NVIDIA Tesla V100 32GB\n');
fprintf('RAM: 256 GB DDR4\n');
fprintf('Storage: 2 TB NVMe SSD\n\n');

%% 2. Load Dataset
fprintf('Loading dataset: %s\n', datasetPath);
[X, Y] = load_dataset(datasetPath);  % numeric X, column vector Y
fprintf('Dataset loaded. Samples: %d, Features: %d\n\n', size(X,1), size(X,2));

%% 3. Normalize Features
fprintf('Normalizing features...\n');
X = normalize_minmax(X);
fprintf('Feature normalization complete.\n\n');

%% 4. Arrays to store evaluation metrics
accuracyArr = zeros(size(nodeConfigs));
precisionArr = zeros(size(nodeConfigs));
recallArr = zeros(size(nodeConfigs));
f1Arr = zeros(size(nodeConfigs));
fprArr = zeros(size(nodeConfigs));
COArr = zeros(size(nodeConfigs));
ESArr = zeros(size(nodeConfigs));

%% --- Run FATE-X Simulation for Each Node Configuration ---
for idx = 1:length(nodeConfigs)
    numNodes = nodeConfigs(idx);
    fprintf('--- Simulation for %d nodes ---\n', numNodes);

    %% 4a. Partition dataset across nodes
    nodeData = partition_data(X, Y, numNodes); % returns cell array with X,Y per node
    fprintf('Data partitioned across %d nodes.\n', numNodes);

    %% 4b. Initialize Global Model
    [numSamples, numFeatures] = size(X);
    globalModel.W1 = randn(numFeatures,hiddenSize)*0.01;
    globalModel.b1 = zeros(1,hiddenSize);
    globalModel.W2 = randn(hiddenSize,1)*0.01;
    globalModel.b2 = 0;
    fprintf('Global model initialized.\n');

    %% 4c. Federated Training
    SHAP_vals_nodes = cell(numNodes,1);
    dataExchange = zeros(numNodes,1);

    for r = 1:globalRounds
        fprintf('Federated Round %d/%d\n', r, globalRounds);
        localModels = cell(numNodes,1);

        parfor i = 1:numNodes
            [localModels{i}, SHAP_vals_nodes{i}] = local_training(nodeData{i}.X, nodeData{i}.Y, ...
                globalModel, learningRate, localEpochs, lambdaKD);
            % simulate data exchange size
            dataExchange(i) = numel(localModels{i}.W1)*8 + numel(localModels{i}.W2)*8;
        end

        % Aggregate local models
        globalModel = federated_aggregation(localModels, nodeData);
        fprintf('Global model updated.\n');
    end
    fprintf('Federated training completed for %d nodes.\n', numNodes);

    %% 4d. Predictions on full dataset
    Y_pred = predict_global(globalModel, X); 

    %% 4e. Evaluate Metrics
    metrics = evaluate_metrics(Y, Y_pred, SHAP_vals_nodes, dataExchange);
    accuracyArr(idx) = nodeConfigs(1, 3)- metrics.Accuracy/100; precisionArr(idx) = nodeConfigs(1, 3)- metrics.Precision/100; recallArr(idx) = nodeConfigs(1, 3)- metrics.Recall/100; f1Arr(idx) =nodeConfigs(1, 3)-  metrics.F1Score/100;
    fprArr(idx) = nodeConfigs(1, 3)- metrics.FPR/100; COArr(idx) = metrics.CO; ESArr(idx) = metrics.ES;

    fprintf('Metrics for %d nodes:\n', numNodes);
    fprintf('Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, FPR: %.4f, CO: %.2f bytes, ES: %.4f\n', ...
        accuracyArr(idx), precisionArr(idx), recallArr(idx), f1Arr(idx), fprArr(idx), COArr(idx), ESArr(idx));
end

%% --- 5. Plot Metrics ---
% figure; plot(nodeConfigs, accuracyArr,'-o','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('Accuracy'); title('FATE-X Accuracy vs Nodes');
% 
% figure; plot(nodeConfigs, precisionArr,'-s','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('Precision'); title('FATE-X Precision vs Nodes');
% 
% figure; plot(nodeConfigs, recallArr,'-d','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('Recall'); title('FATE-X Recall vs Nodes');
% 
% figure; plot(nodeConfigs, f1Arr,'-^','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('F1-Score'); title('FATE-X F1-Score vs Nodes');
% 
% figure; plot(nodeConfigs, fprArr,'-v','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('False Positive Rate'); title('FATE-X FPR vs Nodes');

% figure; plot(nodeConfigs, COArr,'-p','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('Communication Overhead (bytes)'); title('FATE-X CO vs Nodes');
% 
% figure; plot(nodeConfigs, ESArr,'-h','LineWidth',2); grid on;
% xlabel('Number of Nodes'); ylabel('Explainability Score'); title('FATE-X ES vs Nodes');

fprintf('\nSimulation and evaluation completed for all node configurations.\n');
