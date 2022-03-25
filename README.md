# Text Mining with TF-IDF & Cosine Similarity

A simple python repository for developing perceptron based text mining involving dataset linguistics preprocessing for text classification and extracting similar text for a given query.

New Implementation: Added PyTorch based optimization handling buggy loading of sparse 'csr_matrix' to cuda tensor.

## Outcomes

1. Numpy implementation,

    |Vanilla Optimization|Optimization with L2-Regularization|
    |:--:|:--:|
    |<p align="left"><img src="outcomes/Confusion Matrix.png" width="350">|<p align="left"><img src="outcomes/Confusion Matrix with L2R.png" width="350">|

    Top 5 weighted terms,

    |Terms|Weights|Terms - L2|Weights - L2 Regularizded|
    |:--:|:--:|:--:|:--:|
    |langeweile|7.094|top|5.8911|
    |geilo|7.0535|langeweile|5.8396|
    |best|6.7828|geilo|5.7615|
    |love|6.376|perfekt|5.6325|
    |exzellent|6.3534|super|5.6279|

2. PyTorch implementation,

    |Vanilla Optimization|Optimization with L2-Regularization|
    |:--:|:--:|
    |<p align="left"><img src="torch_implementation/data/Non Penalized Prediction.png" width="350">|<p align="left"><img src="torch_implementation/data/Penalized Prediction.png" width="350">|
    |Histogram:Weights|Penalized Weights|
    |<p align="left"><img src="torch_implementation/data/Non Penalized Weights.png" width="350">|<p align="left"><img src="torch_implementation/data/Penalized Weights.png" width="350">|

    Top 5 weighted terms,

    |Terms|Weights|Terms - L2|Weights - L2 Regularizded|
    |:--:|:--:|:--:|:--:|
    |erfolgreichen|20.5452|cool|8.8814|
    |anmeldungen|20.0064|geil|8.0933|
    |angemessene|19.658|super|6.7332|
    |eonfach|19.5906|top|5.4004|
    |verarbeitung|19.5136|gut|4.8924|

## Dependencies

Install dependencies using:

```bash
pip3 install -r requirements.txt 
```

## Contact

* Email: navalekanishk@gmail.com
* Website: <https://kanishknavale.github.io/>
