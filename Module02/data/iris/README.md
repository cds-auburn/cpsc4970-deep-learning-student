# Iris samples

Source: Fisher, R. A. (1936). [Iris](https://doi.org/10.24432/C56C76). UCI Machine
Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Contents |
|---|---|
| `iris.csv` | The copy distributed with scikit-learn 1.6.1 (BSD 3-Clause), downloaded 2026-11-18 from `https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.6.1/sklearn/datasets/data/iris.csv`. This is the file M2-030 loads from the same URL. One header line, then 150 samples. |
| `test.csv` | The 25 samples kept out of training the M2-030 model, written by `tools/train_iris_model.py` in the order they were drawn. One header line of column names, then 25 samples. |

Both files have four measurements in centimeters (sepal length, sepal width,
petal length, petal width) followed by a species code: 0 = setosa,
1 = versicolor, 2 = virginica. Neither file has a row-ID column.

SHA-256 of `iris.csv`: `f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449`
(also recorded in `models/iris/metadata.json`).
