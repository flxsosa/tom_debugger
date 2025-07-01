## AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
### [Paper](https://arxiv.org/abs/2502.15676) | [Project Page](https://chuanyangjin.com/AutoToM) | [Tweet](https://x.com/chuanyang_jin/status/1894737913499246665)

We propose AutoToM, an automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind. 

![intro](visuals/intro.png)

## Example Usage

*To run AutoToM on MMToM-QA, with the default settings of reduced hypotheses and backwards inference*: 

    python ProbSolver.py --automated --dataset_name "MMToM-QA"

*To run AutoToM on ToMi-1st with a specified model input*: 

    python ProbSolver.py --dataset_name "ToMi-1st" --assigned_model "['State', 'Observation', 'Belief']"

## Requirements

- Install relevant packages:
    - run
    ``
        pip install -r requirements.txt
    ``
- Set your `OPENAI_API_KEY`:

    - On macOS and Linux:
    `export OPENAI_API_KEY='your-api-key'`
    
    - On Windows: `set OPENAI_API_KEY='your-api-key'`

## Testing AutoToM with customized questions

Please check out ``playground.ipynb``. Simply replace the story and choices with your customized input to see how *AutoToM* discover Bayesian models and conduct inverse planning!

## Citation

Please cite the paper and star this repo if you find it useful, thanks!

```bibtex
@article{zhang2025autotom,
  title={AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind},
  author={Zhang, Zhining and Jin, Chuanyang and Jia, Mung Yao and Shu, Tianmin},
  journal={arXiv preprint arXiv:2502.15676},
  year={2025}
}
```
