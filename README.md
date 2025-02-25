## AutoToM: Automated Bayesian Inverse Planning and Model Discovery<br>for Open-ended Theory of Mind
### [Paper](https://arxiv.org/abs/2502.15676) | [Project Page](https://chuanyangjin.com/AutoToM)

![intro](visuals/intro.png)

We propose AutoToM, an automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind. 

To run AutoToM with a specified benchmark, with the default settings of reduced hypotheses and backwards inference: 
## 
    $ python ProbSolver.py --automated --dataset_name "MMToM-QA"

To run AutoToM with a specified model input: 
##
    $ python ProbSolver.py --dataset_name ToMi-1st --assigned_model "['State', 'Observation', 'Belief']"

## Files 

## Testing the model with customized questions

Please check out the playground.ipynb
