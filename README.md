# RL-DQAS
This is the repository for the paper [Differentiable Quantum Architecture Search for Quantum Reinforcement Learning(RL-DQAS)](https://arxiv.org/pdf/2309.10392).

Authors: Yize Sun, Yunpu Ma, Volker Tresp

We implement the differentiable quantum architecture search (DQAS) [1] for quantum reinforcment learning (QRL) showing the potential of quantum computing for an industrial application. We apply a gradientbased framework DQAS on reinforcement learning tasks and evaluate it in two different environments - cart pole and frozen lake. It contains input- and output weights, progressive search, and other new features. The experiments conclude that DQAS can design quantum circuits automatically and efficiently. The evaluation results show significant outperformance compared to the manually designed circuit. Furthermore, the performance of the automatically created circuit depends on whether the supercircuit learned well during the training process. This work is the first to show that gradient-based quantum architecture search is applicable to QRL tasks.

Please use the following BibTex for citation:
```
@inproceedings{sun2023differentiable,
  title={Differentiable quantum architecture search for quantum reinforcement learning},
  author={Sun, Yize and Ma, Yunpu and Tresp, Volker},
  booktitle={2023 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  volume={2},
  pages={15--19},
  year={2023},
  organization={IEEE}
}

```

[1]: Zhang, Shi-Xin, et al. "Differentiable quantum architecture search." Quantum Science and Technology 7.4 (2022): 045023.
