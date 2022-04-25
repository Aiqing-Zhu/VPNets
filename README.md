# PyTorch Implementation of: VPNets: Volume-preserving neural networks for learning source-free dynamics

## Requirements 
* Python 
* torch
* numpy
* matplotlib

## Reproducing the results of the paper
In general all parameters which need to be specified are given in the paper.

### Running Experiments Volterra equations:
To train the models, run:
```
python LV.py --filename 'lv-l1' ----net_type 'LA'
python LV.py --filename 'lv-g1' ----net_type 'G'
```
After training, run:
```
python LV_output.py
```
on CPU


### Running Experiments Charged particle dynamics:
Here,  we used 5 different seed which can also be set via the command line `random_seed` parameter.
To train the models, run:
```
python LF.py --filename 'lf-l'   --lr 0.01 --iterations 800000 
python LF.py --filename 'lf-g'   --lr 0.001 --iterations 500000 
```
After training, run:
```
python LF_output.py
```
on CPU


## References
[1] [learner](https://github.com/jpzxshi/learner)
