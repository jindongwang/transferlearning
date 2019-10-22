function [Xs_new] = CORAL_map(Xs,Xt)
	cov_src = cov(Xs) + eye(size(Xs,2));
	cov_tar = cov(Xt) + eye(size(Xt,2));
	A_coral = cov_src^(-1/2) * cov_tar^(1/2);
	Xs_new = Xs * A_coral;
end