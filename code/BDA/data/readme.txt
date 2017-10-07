This is the widely used Office+Caltech dataset firstly released by Boqing Gong.
When use it, please perform scaling after loading the dataset.
Example:
		fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    	Xt = zscore(fts,1);fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
    	Xt = zscore(fts,1);

If you are tired of transforming features, you can use the data from zscore folder.