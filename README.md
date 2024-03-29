
The code presented is used in two publications
([preprint here](https://arxiv.org/abs/2111.06680)
and [here](https://arxiv.org/abs/2304.09488)).

[1] S. Gracla, E. Beck, C. Bockelmann and A. Dekorsy,
"Deep Reinforcement Model Selection for Communications Resource Allocation in On-Site Medical Care",
in *Proc. 2022 IEEE Wireless Communications and Networking Conference (WCNC)*,
Austin/Online, USA, 13. - 16. April 2022,
pp. 1485-1490, doi: [10.1109/WCNC51071.2022.9771679](https://doi.org/10.1109/WCNC51071.2022.9771679).

[2] S. Gracla, E. Beck, C. Bockelmann and A. Dekorsy,
"Learning Resource Scheduling with High Priority Users using Deep Deterministic Policy Gradients",
in *Proc. ICC 2022 - IEEE International Conference on Communications*,
Seoul/Online, South Korea, 16. - 20. May 2022,
pp. 4480-4485, doi: [10.1109/ICC45855.2022.9838349](https://doi.org/10.1109/ICC45855.2022.9838349).

Email: {**gracla**, beck, bockelmann, dekorsy}@ant.uni-bremen.de

The `scheduling` folder contains the code for [1], while `scheduling_policygradient` contains the code for [2].

The structure is as follows:

```
/
├─ [proj_name]/         | project folder
│  ├─ imports/          | contains python modules for import
│  ├─ *_config.py       | contains config for this project
│  ├─ *_runner.py       | orchestrates training and testing
│  ├─ *_test.py         | wrappers for testing a trained scheduler
│  ├─ *_train.py        | wrappers for training a scheduler
├─ .gitignore           | .gitignore
├─ README.md            | this file
├─ requirements.txt     | python packages & versions used
```

