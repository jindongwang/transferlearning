## SOT (Substructural Optimal Transport)

This is the implementation of SOT in Python.

### Usage

4 hyper-paramters need to be carefully set. $d$ means the cluster numbers of the target data. $reg_e$ corresponds to $\lambda$, $reg_{cl}$ corresponds to $\eta$, and $reg_{ce}$ corresponds to $\lambda_1$ in the origin paper.

### Demo

I offer a basic demo to run on the datasets used in the paper. Download the test data [HERE](https://pan.baidu.com/s/1m-lkCklSWSreuEeBfp9GPQ). Extraction code is ```1zk9```. Run
```python main.py```

### Results

SOT achieved **state-of-the-art** performances compared to a lot of traditional methods. The following results are from the original paper.


|         | Method    | D - H | D - U | D - P    | H - D | H - U | H-P   |  U - D | U - H | U - P    | P - D | P - H | P-U   | AVG   |
|---------|-----------|-------|-------|--------|-------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
|   | 1NN | 62.70 | 56.40 | 66.03 | 65.16 | 55.03 | 60.81 | 71.30 | 61.38 | 60.37 | 60.99 | 55.26 | 51.63 | 60.59 |
|   | LMNN  | 55.24|65.74|48.38|64.27|56.55|65.00|64.58|65.46|67.13|59.53|39.11|41.21|57.68  |
| | TCA       | 60.79|51.98|65.66|62.50|41.87|52.06|69.06|53.43|60.88|57.81|46.38|51.45|56.16  |
| ICCV 2013 | SA      | 63.61|57.36|65.44|66.35|55.60|60.88|70.62|59.64|61.18|62.60|55.45|50.58|60.78 |
| AAAI 2016| CORAL      | 64.23|52.25|64.85|64.48|53.03|64.41|68.75|61.93|60.15|60.21|56.20|54.46|60.41 |
| Percom 2018 | STL       | 62.83|70.93|65.66|66.15|65.89|67.43|74.69|68.76|65.00|68.96|56.75|55.27|65.69  |
| | OT       | 62.13|65.86|65.66|68.91|58.58|67.50|69.90|59.49|63.75|66.77|51.71|57.59|63.15|
| TPAMI 2016 | OTDA   | 59.36|54.97|65.52|68.91|59.45|67.50|70.26|62.25|63.09|67.19|53.41|59.09|62.58  |
| IJCAL 2020 | MLOT   |62.53|53.33|64.85|68.12|58.10|62.13|69.53|59.68|61.25|65.99|63.30|49.24|61.51  |
| | SOT | **67.74**|**79.74**|**73.31**|**73.39**|**70.87**|**73.23**|**80.99**|**78.04**|**74.41**|**76.46**|**72.82**|**76.51**|**74.79** |


### Reference

If you use this code, please cite it as:

`
Lu W, Chen Y, Wang J, et al. Cross-domain Activity Recognition via Substructural Optimal Transport[J]. arXiv preprint arXiv:2102.03353, 2021.
`

Or in bibtex style:

```
@article{lu2021cross,
  title={Cross-domain Activity Recognition via Substructural Optimal Transport},
  author={Lu, Wang and Chen, Yiqiang and Wang, Jindong and Qin, Xin},
  journal={arXiv preprint arXiv:2102.03353},
  year={2021}
}
```
