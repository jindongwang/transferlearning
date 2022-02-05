import scipy.io
import scipy.stats
import numpy as np
from EasyTL import EasyTL
import time

if __name__ == "__main__":
	datadir = r"D:\Datasets\EasyTL\amazon_review"
	str_domain = ["books", "dvd", "elec", "kitchen"]
	list_acc = []
	
	for i in range(len(str_domain)):
		for j in range(len(str_domain)):
			if i == j:
				continue
			
			print("{} - {}".format(str_domain[i], str_domain[j]))
			
			mat1 = scipy.io.loadmat(datadir + "/{}_400.mat".format(str_domain[i]))
			Xs = mat1["fts"]
			Ys = mat1["labels"]
			mat2 = scipy.io.loadmat(datadir + "/{}_400.mat".format(str_domain[j]))
			Xt = mat2["fts"]
			Yt = mat2["labels"]
			
			Ys += 1
			Yt += 1
            
			Xs = Xs / np.tile(np.sum(Xs,axis=1).reshape(-1,1), [1, Xs.shape[1]])
			Xs = scipy.stats.mstats.zscore(Xs);
			Xt = Xt / np.tile(np.sum(Xt,axis=1).reshape(-1,1), [1, Xt.shape[1]])
			Xt = scipy.stats.mstats.zscore(Xt);
			
			Xs[np.isnan(Xs)] = 0
			Xt[np.isnan(Xt)] = 0
            
			t0 = time.time()
			Acc1, _ = EasyTL(Xs,Ys,Xt,Yt,"raw")
			t1 = time.time()
			print("Time Elapsed: {:.2f} sec".format(t1 - t0))
			Acc2, _ = EasyTL(Xs,Ys,Xt,Yt)
			t2 = time.time()
			print("Time Elapsed: {:.2f} sec".format(t2 - t1))
            
			print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1*100, Acc2*100))
			list_acc.append([Acc1,Acc2])
            
	acc = np.array(list_acc)
	avg = np.mean(acc, axis=0)
	print('EasyTL(c) AVG Acc: {:.1f} %'.format(avg[0]*100))
	print('EasyTL AVG Acc: {:.1f} %'.format(avg[1]*100))