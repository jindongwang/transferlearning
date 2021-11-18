git clone https://github.com/facebookresearch/DomainBed
cd DomainBed/
git checkout 2deb150
python3 -m pip install gdown==3.13.0
# python3 -m pip install wilds==1.1.0 torch_scatter # Installing `torch_scatter` seems quite involved. Turn to edit the files to exclude the import.
vi "+14norm I# " "+15norm I# " "+271norm I# " "+267s/# //" "+270s/# //" "+wq" domainbed/scripts/download.py
vi "+12norm I# " "+13norm I# " "+wq" domainbed/datasets.py
python3 -m domainbed.scripts.download --data_dir=./domainbed/data

