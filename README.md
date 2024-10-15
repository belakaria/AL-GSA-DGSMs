# Active Learning for Derivative-Based Global Sensitivity Analysis with Gaussian Processes

This repository includes all the methods proposed and discussed in the paper [**Active Learning for Derivative-Based Global Sensitivity Analysis with Gaussian Processes**](link). 

Below are steps to run the code: 

### Set up enviroment
```
conda create -y --name gsa python=3.8
conda run -n gsa pip install mpmath 
conda run -n gsa pip install ax-platform 
conda activate gsa
```
### Launch one run with a single seed:
```
mkdir results/function_name
python run.py function_name method_name num_iter num_init seed
```
### Launch a multi-runs experiment with several seeds:
```
bash experiments.sh function_name method_name num_iter num_init repeats
```

- **List of implemented methods names:**
    - `Sobol`: Quasi random Monte Carlo Sequence ($QR$).
    - `MaxVariance`: GP variance ($fVAR$).
    - `InformationGain`: GP information gain ($fIG$).

    - `DerivVarianceTrace`: Variance of the derivative GP ($DV$).
    - `DerivVarianceTraceReduction`: Variance reduction of the derivative GP ($DV_r$).
    - `DerivSumInformationGain`: Information gain of the derivative GP ($DIG$).
    
    - `DerivAbsVarianceTrace`: Variance of the absolute of the derivative GP ($DAbV$).
    - `DerivAbsVarianceTraceReduction`: Variance reduction of the absolute of the derivative GP ($DAbV_r$).
    - `DerivAbsSumInformationGain{i}`: Information gain of the absolute of the derivative GP using the $i$ th approximation ($DAbIG_i$).

    - `DerivSquareVarianceTrace`: Variance of the square of the derivative GP ($DSqV$).
    - `DerivSquareVarianceTraceReduction`: Variance reduction of the square of the derivative GP ($DSqV_r$).
    - `DerivSquareSumInformationGain{i}`: Information gain of the square of the derivative GP using the $i$ th approximation ($DSqIG_i$).

    - `GlobalDerivVarianceTraceReduction`: Global (integrated) variance reduction of the derivative GP ($GDV_r$).
    - `GlobalDerivSumInformationGain`: Global (integrated) information gain of the derivative GP ($GDIG$).
    - `GlobalDerivAbsVarianceTraceReduction`: Global (integrated) variance reduction of the absolute of the derivative GP ($GDAbV_r$).
    - `GlobalDerivAbsSumInformationGain{i}`: Global (integrated) information gain of the square of the derivative GP using the $i$ th approximation ($GDAbIG_i$).
    - `GlobalDerivSquareVarianceTraceReduction`: Global (integrated) variance reduction of the square of the derivative GP ($GDSqV_r$).
    - `GlobalDerivSquareSumInformationGain{i}`: Global (integrated) information gain of the square of the derivative GP using the $i$ th approximation ($GDSqIG_i$).

- **Test functions:** The list of implemented function names can be found in `utils/util.py` and their corresponding true DGSMs are in `utils/dgsm_values.py`

- **Note:** This repository also includes an interface to call the acquisition functions proposed [Wycoff et al 2021](https://arxiv.org/abs/1907.11572). The methods names for these acquisition functions are `activegp:Cvar`, `activegp:Cvar2` and `activegp:Ctr`. For these approaches to work, the requirements [in corresponding R package](https://cran.r-project.org/web/packages/activegp/index.html) have to be installed. 



### Citation
If you use this code, please consider citing our paper:
```bibtex
@article{Belakaria_Letham_2024, 
  title={Active Learning for Derivative-Based Global Sensitivity Analysis with Gaussian Processes}, 
  url={},
  journal={Advances in neural information processing systems}, 
  author={Belakaria, Syrine and Letham, Benjamin and  Doppa, Janardhan Rao and Engelhardt, Barbara and Ermon, Stefano and Bakshy, Eytan}, 
  year={2024}, 
  month={June}}
````