# ICL-Router: In-Context Learned Model Representations for LLM Routing

This repository contains the code and dataset for the  paper:

**"ICL-Router: In-Context Learned Model Representations for LLM Routing"**

![icl_router_framework](/Users/wangchenxu/文件管理/college_course/router/icl_router/figures/icl_router_framework.png)

## Data

All datasets are located in the *./data* directory. Below, we provide a brief description of each file:

- *question_train.json* and *question_test.json*: These files contain the training and test sets, respectively, used for **Query Reconstruction Training**.
- *train_router.json* and *test_router.json*: These files contain the training and test sets, respectively, used for **ICL Model Routing Training**.
- *expert100.json*, *expert300.json*, *expert500.json*  and *expert1000.json*: These files record the performance of each candidate LLM on a representative set of queries, indicating whether each response is correct. The numbers (100, 300, etc.) refer to the number of questions included in each evaluation set. Each set is built by sampling, from each in-domain benchmark, questions that only a few models answered correctly, ensuring a challenging and discriminative evaluation.

## Installation

Ensure you have all dependencies installed by running:

```bash
pip install -r requirements.txt
```

### Query Reconstruction Training

```bash
# Multi-GPU: 8-GPU Training
./scripts/train_stage1.sh 0,1,2,3,4,5,6,7,8 
```

### ICL Model Routing Training

```bash
# Multi-GPU: 8-GPU Training
./scripts/train_stage2.sh 0,1,2,3,4,5,6,7,8 
```

We have already integrated the evaluation code into the training file. In the future, we will also reorganize the code and provide a separate evaluation file.