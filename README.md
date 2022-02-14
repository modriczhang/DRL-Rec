
## DRL-Rec

Reinforcement learning (RL) has been verified in real-world list-wise recommendation. 
However, RL-based recommendation suffers from huge memory and computation costs due to its large-scale models. 

Knowledge distillation (KD) is an effective approach for model compression widely used in practice. 
However, RL-based models strongly rely on sufficient explorations on the enormous user-item space due to the data sparsity issue, which multiplies the challenges of KD with RL models. 

What the teacher should teach and how much the student should learn from each lesson need to be carefully designed. 

In this work, we propose a novel Distilled reinforcement learning framework for recommendation (DRL-Rec), which aims to improve both effectiveness and efficiency in list- wise recommendation. 

Specifically, we propose an Exploring and filtering module before the distillation, which decides what lessons the teacher should teach from both teachers’ and students’ aspects. 

We also conduct a Confidence-guided distillation at both output and intermediate levels with a list-wise KL divergence loss and a Hint loss, which aims to understand how much the student should learn for each lesson. 

DRL-Rec has been deployed on WeChat Top Stories for more than six months, affecting millions of users.

### Requirements:
- Python 3.9
- Tensorflow 2.5.0-rc0

## Note

In the actual online system, DRL-Rec is a complex re-ranking framework implemented in C++. 
All models are trained based on a deeply customized version of distributed tensorflow supporting large-scale sparse features.

Without massive data and machine resources, training DRL-Rec is not realistic.

Therefore, the open source code here only implements a simplified version for interested researchers. If there are any errors, please contact me. Thanks!

## About

"Explore, Filter and Distill: Distilled Reinforcement Learning in Recommendation" ([CIKM 2021](https://dl.acm.org/doi/10.1145/3459637.3481917))
