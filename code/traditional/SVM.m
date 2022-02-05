function [acc,y_pred,time_pass] = SVM(Xs,Ys,Xt,Yt)
    Xs = double(Xs);
    Xt = double(Yt);
    time_start = clock();
    [acc,y_pred] = LinAccuracy(Xs,Xt,Ys,Yt);
    time_end = clock();
    time_pass = etime(time_end,time_start);
end

function [acc,predicted_label] = LinAccuracy(trainset,testset,trainlbl,testlbl)
    model = trainSVM_Model(trainset,trainlbl);
    [predicted_label, accuracy, decision_values] = svmpredict(testlbl, testset, model);
    acc = accuracy(1,1);
end

function svmmodel = trainSVM_Model(trainset,trainlbl)
    C = [0.001 0.01 0.1 1.0 10 100 ];
    parfor i = 1 :size(C,2)
        model(i) = libsvmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q -v 2',C(i) ));
    end
    [val indx]=max(model);
    CVal = C(indx);
    svmmodel = libsvmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q',CVal));
end