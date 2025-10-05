function [X,Y] = load_dataset(path)
T = readtable(path);
X = table2array(T(:,1:end-1));
Y = table2array(T(:,end));
if size(Y,2)>1
    Y = Y(:);
end
end
