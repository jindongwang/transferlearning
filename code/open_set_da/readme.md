# Open Set Domain Adaptation

This directory contains the implementation of the ICCV-17 paper *[Open Set Domain Adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)*. 

## About the code

I'm not the authors of that paper. So this is not official. We all hope the authors can provide the implementations ASAP. However, according to them, the code is not available for now since they developed this method in a company, there is some rules to follow.

The official repo of this paper is: [Heliot7/open-set-da](https://github.com/Heliot7/open-set-da). We hope authors can get publish the code soon.

## About the method

The method in the paper contains 2 main processes: 
- Get label assignment using Eq.(1)
- And learn the mapping $W$

Due to some reasons, part 2 is hard to understand. So I could only re-implement the part 1, which is basically an integer programming. I hope to get more information about part 2, so I will add this soon.

The code is written in Matlab.

## Get label assignment

The code is just in `get_label_binary.m`. For the meaning of this function, please refer to Eq.(1) of the original paper.

## Get the mapping $W$

To be added.

## My understanding (in Chinese)

Since this paper is quite original and has obtain good results, I've also provided an article reading about this paper. Please refer to the [article](https://zhuanlan.zhihu.com/p/31230331) (in Chinese).

## Reference

Pau Panareda Busto, Juergen Gall. Open set domain adaptation. ICCV 2017.