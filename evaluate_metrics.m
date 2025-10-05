function metrics = evaluate_metrics(Y_true,Y_pred,SHAP_vals_nodes,dataExchange)
TP = sum((Y_pred==1)&(Y_true==1));
TN = sum((Y_pred==0)&(Y_true==0));
FP = sum((Y_pred==1)&(Y_true==0));
FN = sum((Y_pred==0)&(Y_true==1));

metrics.Accuracy = (TP+TN)/(TP+TN+FP+FN);
metrics.Precision = TP/(TP+FP+eps);
metrics.Recall = TP/(TP+FN+eps);
metrics.F1Score = 2*(metrics.Precision*metrics.Recall)/(metrics.Precision+metrics.Recall+eps);
metrics.FPR = FP/(FP+TN+eps);
metrics.CO = sum(dataExchange);

% Explainability score (average of all SHAP placeholders)
allSHAP = cell2mat(SHAP_vals_nodes);
metrics.ES = mean(allSHAP);
end
