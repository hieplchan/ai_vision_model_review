# 1. Training parameters
## 1.1 SGD params

|          Parameters          |   Value   |     Type     |              Note              |
| :--------------------------- | :------:  | :----------: | :----------------------------: |
| learning-rate                | 1e-3      | float        |  |
| momentum                     | 0.9       | float        |  |
| weight_decay                 | 5e-4      | float        |  |
| gamma                        | 0.1       | float        |      Gamma update for SGD      |
| extra_layers_lr              | None      | float        |  |

## 1.2 Scheduler params

|          Parameters          |    Value    |     Type     |              Note              |
| :--------------------------- | :--------:  | :----------: | :----------------------------- |
| scheduler                    | multi-step  | string       | one of multi-step and cosine   |
| milestones                   | 80,100      | string       | for MultiStepLR                |
| t_max                        | 120         | float        | for Cosine Annealing Scheduler |

# 2. Dataset
## 2.1 COCO
[Format](https://www.youtube.com/watch?v=h6s61a_pqfM)  
