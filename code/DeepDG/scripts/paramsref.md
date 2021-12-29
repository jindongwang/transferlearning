## We offer a hyperparameter reference for our implementations.

### Common hyperparameters
* max_epoch:120
* lr: 0.001 or 0.005

### Hyperparameters for PACS(ResNet-18) (Corresponding to each task in order, ACPS)

| Method | A | C | P | S |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.5| 1| 0.1| 0.1|
|Mixup(mixupalpha)|0.1| 0.2| 0.2| 0.2|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.3|0.3,0.1|0.1,0.1|0.1,0.1|
|MMD(mmd_gamma)|10|1|0.5|0.5|
|CORAL(mmd_gamma)|0.5|0.1|1|0.01|
|GroupDRO(groudro_eta)|10**(-2.5)| 0.001| 0.001| 0.01|
|ANDMask(tau)|0.5|0.82|0.5|0.5|
|VREx(lam,anneal_iters)|1,5000|1,100|0.3,5000|1,10|

### Hyperparameters for Office-Home(ResNet-18) (Corresponding to each task in order, ACPR)

| Method | A | C | P | R |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.1| 0.1| 0.1| 0.1|
|Mixup(mixupalpha)|0.2| 0.2| 0.1| 0.1|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.1|0.1,0.1|0.1,0.3|0.1,0.1|
|MMD(mmd_gamma)|0.5|10|0.5|0.01|
|CORAL(mmd_gamma)|1|1|0.5|0.01|
|GroupDRO(groudro_eta)|0.001| 10**(-2.5)| 10**(-2.5)| 0.01|
|ANDMask(tau)|0.82|0.82|0.82|0.82|
|VREx(lam,anneal_iters)|0.3,500|10,100|10,100|10,100|

### Hyperparameters for PACS(ResNet-50) (Corresponding to each task in order, ACPS)

| Method | A | C | P | S |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.1| 0.1| 0.1| 1|
|Mixup(mixupalpha)|0.1| 0.1| 0.2| 0.1|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.3|0.3,0.3|0.1,0.1|0.3,0.3|
|MMD(mmd_gamma)|0.1|1|1|0.5|
|CORAL(mmd_gamma)|0.01|1|0.01|1|
|GroupDRO(groudro_eta)|0.001| 10**(-2.5)| 0.001| 0.1|
|ANDMask(tau)|0.82|0.5|0.82|0.82|
|VREx(lam,anneal_iters)|1,500|0.3,5000|0.3,5000|0.3,500|

### Hyperparameters for Office-Home(ResNet-50) (Corresponding to each task in order, ACPR)

| Method | A | C | P | R |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.1| 0.1| 0.1| 0.1|
|Mixup(mixupalpha)|0.1| 0.2| 0.2| 0.1|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.3|0.3,0.1|0.1,0.1|0.3,0.1|
|MMD(mmd_gamma)|1|0.5|0.5|1|
|CORAL(mmd_gamma)|0.1|1|0.5|0.01|
|GroupDRO(groudro_eta)|0.01| 0.1| 10**(-2.5)| 10**(-2.5)|
|ANDMask(tau)|0.5|0.5|0.82|0.5|
|VREx(lam,anneal_iters)|0.3,100|10,500|0.3,100|10,500|

### Remark

Environments may affect results.

You can try to adjust the parameters by yourself or just use the same environment following (https://hub.docker.com/r/jindongwang/docker).

