# DualBreach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[NDSS 2026] [DualBreach: Efficient Dual-Jailbreaking via Target-Driven Initialization and Multi-Target Optimization](https://arxiv.org/abs/2504.18564)".



## Abstract

Recent research has focused on exploring the vulnerabilities of Large Language Models (LLMs), aiming to elicit harmful and/or sensitive content from LLMs. However, due to the insufficient research on dual-jailbreaking---attacks targeting both LLMs and Guardrails, the effectiveness of existing attacks is limited when attempting to bypass safety-aligned LLMs shielded by guardrails. Therefore, in this paper, we propose DualBreach, a target-driven framework for dual-jailbreaking. DualBreach employs a \textit{Target-driven Initialization} (TDI)  strategy to dynamically construct initial prompts, combined with a \textit{Multi-Target Optimization} (MTO) method that utilizes approximate gradients to jointly adapt the prompts across guardrails and LLMs, which can simultaneously save the number of queries and achieve a high dual-jailbreaking success rate.  For black-box guardrails, DualBreach either employs a powerful open-sourced guardrail or imitates the target black-box guardrail by training a proxy model, to incorporate guardrails into the MTO process.

We demonstrate the effectiveness of DualBreach in dual-jailbreaking scenarios through extensive evaluation on several widely-used datasets. Experimental results indicate that DualBreach outperforms state-of-the-art methods with fewer queries, achieving significantly higher success rates across all settings. More specifically, DualBreach achieves an average dual-jailbreaking success rate of 93.67\% against GPT-4 with Llama-Guard-3 protection, whereas the best success rate achieved by other methods is 88.33\%. Moreover, DualBreach only uses an average of 1.77 queries per successful dual-jailbreak, outperforming other state-of-the-art methods. For defense, we propose an XGBoost-based ensemble defensive mechanism named EGuard, which integrates the strengths of multiple guardrails, demonstrating superior performance compared with Llama-Guard-3.











**Figure 1.** Examples of different jailbreaking scenarios. 
![Jailbreak example](figs/diverse_example.png)

**Figure 2.** Workflow of DualBreach framework.  
![Workflow](figs/workflow.png)

**Figure 3.** EGuard.  
![EGuard](figs/EGuard.png)

**Figure 4.** Example of DualBreach.  
![strongreject_gpt4o](figs/strongreject_gpt4o.png)





## Code

### Preparation

Befor we start, we need to download the target LLMs you want to jailbreak and the dataset you want to use. We list the six target LLMs and two benchmark datasets used in our paper below:



**Datasets** </br>
| Dataset      | Link                                                         |
|-----------   |--------------------------------------------------------------|
| AdvBench     | https://github.com/llm-attacks/llm-attacks                   |
| DNA     | https://github.com/llm-attacks/llm-attacks                   |
| HarmBench    | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |



After downloading three benchmark datasets, you need to put them into ``./data`` folder (if not exist, create a folder with the same name).




### Code
**1) Download this GitHub**
```
git clone https://github.com/hxz-sec/DualBreach.git
```

**2) Setup Environment**

We recommend conda for setting up a reproducible experiment environment.
We include `environment.yaml` for creating a working environment:

```bash
conda env create -f environment.yaml -n DualBreach
```

Then use requirements.txt or manually install pip in the project:
```
pip install -r requirements.txt
```

Run the main code :

```bash
conda activate DualBreach
```



**3) Train proxy model** (optionl)


```
python /proxy_model/train/google_train.py
python /proxy_model/train/openai_train.py 
```


**4) Run DualBreach** 


```
bash   guard_attack.sh
```

Evaluate:
```
python eval/response_eval/response_eval.py
python eval/response_eval/attack_success_evaluate.py
```


**5) EGuard** 

Train guard model based on Xgboost
```
python train_xgboost.py 
python evaluate_xgboost.py 
```


## Citation
If you find this useful in your research, please consider citing:

NDSS 2026 Submission (preferred)

```
@article{huang2025DualBreach,
  title={DualBreach: Efficient Dual-Jailbreaking via Target-Driven Initialization and Multi-Target Optimization},
  author={Huang, Xinzhe and Xiu, Kedong and Zheng, Tianhang and Zeng, Churui and Ni, Wangze and Qin, Zhan and Ren, Kui and Chen, Chun},
  journal={arXiv preprint arXiv:2504.18564},
  year={2025}
}


```

## Lisence
This project is licensed under the MIT License.
