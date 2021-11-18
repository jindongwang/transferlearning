python3 makedata.py train --pleft 1. 0. --distr randn --loc 5. --scale 1.
python3 makedata.py test  --pleft .5 .5 --distr randn --loc 0. --scale 0.
mv ./data/MNIST/processed/test01_0.5_0.5_randn_0.0_0.0.pt ./data/MNIST/processed/test_01.pt
python3 makedata.py test  --pleft .5 .5 --distr randn --loc 0. --scale 2.
