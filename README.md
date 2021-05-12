# MTRec

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-2.0+-blue'>
  <img src='https://img.shields.io/badge/pipy-v0.0.1-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
</p>

MTRec is a simple **multi-task recommendation** package based Tensorflow2.x. You can install it to invoke some classic models, such as MMoE, and we also provide a `test file` to help you better use it.

Let's get started!



## Installation in Python

MTRec is on PyPI, so you can use `pip` to install it.

```
pip install mtrec==0.0.1

pip install mtrec
```

The default dependency that MTRec has are Tensorflow2.x. 



## Example

There are some [simple tests](test) to use mtrec package.

**First，Dataset**. 

1. MTREC stipulates that features must be sparse discrete features, and continuous features need discrete buckets.
2. The dataset is output in the form of a dictionary, for example`{'name1':[n1,n2,...], 'name2':[n1,n2,...]}`

**Second, Build Model.**

```python
from mtrec import MMoE

task_names = ['task1', 'task2']
num_experts = 3

model = MMoE(task_names, num_experts, sparse_feature_columns)
```

for `sparse_feature_columns`,

```python
from mtrec.functions.feature_column import sparseFeature

embed_dim = 4

sparse_feature_columns = [sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim) for feat in sparse_features]
```

see the `test/utils.py` file for specific details.

**Third, Compile, Fit and Predict**

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

model.compile(loss={'task1': 'binary_crossentropy', 'task2': 'binary_crossentropy'},
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])

model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
    )

pred = model.predict(test_X, batch_size=batch_size)
```



## Model

| Model |                            Paper                             | Published | Author/Group |
| :---: | :----------------------------------------------------------: | :-------: | :----------: |
| MMoE  | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | KDD, 2018 |    Google    |



## Discussion

1. If you have any suggestions or questions about the project, you can leave a comment on `Issue` or email `zggzy1996@163.com`.
2. wechat：

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="25%"/></div>