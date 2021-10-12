## We offer a hyperparameter reference for our method.

### Common hyperparameters
* max_epoch:120
* lr: 0.001 or 0.005

### Hyperparameters for PACS (Corresponding to each task in order, ACPS)

| Method | A | C | P | S |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.1| 0.1| 0.1| 1|
|Mixup(mixupalpha)|0.2| 0.1| 0.2| 0.1|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.3|0.3,0.3|0.1,0.1|0.1,0.3|
|CORAL(mmd_gamma)|1|1|1|1|
|GroupDRO(groudro_eta)|10**(-1.5)| 10**(-2.5)| 10**(-1.5)| 0.1|


### Hyperparameters for Office-Home (Corresponding to each task in order, ACPR)

| Method | A | C | P | R |
|----------|----------|----------|----------|----------|
|DANN(alpha)|0.1| 1| 0.5| 0.1|
|Mixup(mixupalpha)|0.1| 0.1| 0.2| 0.2|
|RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.3,0.1|0.1,0.1|0.3,0.1|0.1,0.1|
|CORAL(mmd_gamma)|0.1|0.5|1|1|
|GroupDRO(groudro_eta)|0.001| 10**(-2.5)| 0.001| 0.001|

### Remark

Environments may affect results.

You can try to adjust the parameters by yourself or just use the same environment following (https://hub.docker.com/r/jindongwang/docker).

