%% Subspace Alignment  ICCV-13

% Inputs 
% Source_Data : normalized source data. Use NormalizeData function to
% normalize the data before
%
% Target_Data :normalized target data. Use NormalizeData function to
% normalize the data before
% 
% Xs : source eigenvectors obtained from normalized source data (e.g. PCA)
% Xt : target eigenvectors obtained from normalized source data (e.g. PCA)
% 
% Source_label : source class label
% Target_label : target class label
%
%
function [acc,y_pred,time_pass] =  SA_SVM(Source_Data,Source_label,Target_Data,Target_label,Xs,Xt)

% Subspace alignment and projections

% NN_Neighbours = 1; %  neares neighbour classifier
% predicted_Label = cvKnn(Target_Projected_Data', Target_Aligned_Source_Data', Source_label, NN_Neighbours);        
% r=find(predicted_Label==Target_label);
% accuracy_sa_nn = length(r)/length(Target_label)*100; 
% 
% NN_Neighbours = 1; %  neares neighbour classifier
% predicted_Label = cvKnn(Target_Data', Source_Data', Source_label, NN_Neighbours);        
% r=find(predicted_Label==Target_label);
% accuracy_na_nn = length(r)/length(Target_label)*100; 

time_start = clock();
A = (Xs*Xs')*(Xt*Xt');
Sim = Source_Data * A *  Target_Data';
[acc,y_pred] = SVM_Accuracy (Source_Data, A,Target_label,Sim,Source_label);
time_end = clock();
% accuracy_na_svm = LinAccuracy(Source_Data,Target_Data,Source_label,Target_label)	;
time_pass = etime(time_end,time_start);

end

function Data = NormalizeData(Data)
    Data = Data ./ repmat(sum(Data,2),1,size(Data,2)); 
    Data = zscore(Data,1);  
end


function [res,predicted_label] = SVM_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
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
	
	model = libsvmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -q',CVal));
	[predicted_label, accuracy, decision_values] = svmpredict(testlabelsref, Sim, model);
	res = accuracy(1,1);
end


function acc = LinAccuracy(trainset,testset,trainlbl,testlbl)	           
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