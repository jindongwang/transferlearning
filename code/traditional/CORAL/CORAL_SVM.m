function [acc,y_pred,time_pass] = CORAL_SVM(Xs,Ys,Xt,Yt)
%% This combines CORAL and SVM. Very simple, very easy to use.
%% Please download libsvm and add it to Matlab path before using SVM.

    time_start = clock();
    %CORAL
    Xs = double(Xs);
    Xt = double(Xt);
    Ys = double(Ys);
    Yt = double(Yt);
    cov_source = cov(Xs) + eye(size(Xs, 2));
    cov_target = cov(Xt) + eye(size(Xt, 2));
    A_coral = cov_source^(-1/2)*cov_target^(1/2);
    Sim_coral = double(Xs * A_coral * Xt');
    [acc,y_pred] = SVM_Accuracy(double(Xs), A_coral, double(Yt), Sim_coral, double(Ys));
    time_end = clock();
    time_pass = etime(time_end,time_start);
end

function [res,predicted_label] = SVM_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
    % Using Libsvm
    Sim_Trn = trainset * M *  trainset';
    index = [1:1:size(Sim,1)]';
    Sim = [[1:1:size(Sim,2)]' Sim'];
    Sim_Trn = [index Sim_Trn ];

    C = [0.001 0.01 0.1 1.0 10 100 1000 10000];
    parfor i = 1 :size(C,2)
        model(i) = libsvmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -v 2 -q',C(i)));
    end
    [val indx]=max(model);
    CVal = C(indx);
    model = svmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -q',CVal));
    [predicted_label, accuracy, decision_values] = svmpredict(testlabelsref, Sim, model);
    res = accuracy(1,1);
end