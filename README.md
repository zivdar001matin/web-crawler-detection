# Web Crawler Detection
## Introduction
Machine Learning hybrid approach for detecting web crawlers (anomaly detection). Used MLflow in order to run as a web service. This is final project of [Rahnema College](https://rahnemacollege.com/) machine learning internship. Data used to fit the model was given by [Sanjagh](https://sanjagh.pro/) server log. In order to prevent malicious usages in future, we don't publish the data but you can use generated server log for fitting the model.
### Built With
* [MLflow](https://www.mlflow.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [scikit-learn](https://scikit-learn.org/)

## Getting Started
The project is written used by `sklearn.pipeline.Pipeline()`. `LogTransformer()` preprocesses log data, `PCAEstimator()` predict using pca and then finally predict output using `RuleBasedEstimator()` by written rules.
These are the steps to run the project on local machine.
### Prerequisites
In order to install requirements using `pip`, run this command:
```bash
$ pip install -r requirements.txt
```

### Running the Code
To fit the pca run this command:

(to fit autoencoder you can replace `pca.py` with `autoencoder.py`)
```bash
$ python pca.py
```
in order to run the model's API on local host: 
(If you don't have the required modules for the file and would like to create a conda environment, remove the argument `--no-conda`.)
```bash
$ mlflow models serve -m mlruns/0/MODEL_RUN_ID/artifacts/model/ -p 8000 --no-conda
```
or you can use pretrained model using real data:
```bash
$ mlflow models serve -m mlruns/0/2ec85ec6bfa74757835225e334311a3e/artifacts/model/ -p 8000 --no-conda
```
### Send Request
You can use other formats in order to send data, see also [Deploy MLflow Models](https://www.mlflow.org/docs/latest/models.html#deploy-mlflow-models).
```bash
$ curl	--location --request POST '127.0.0.1:8000/invocations' \
	--header 'Content-Type: text/csv' \
	--data-binary './datasets/test.csv'
```

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/MLflow_API_request.gif)

## Project Description
Anomaly detection is one of the most popular machine learning techniques. In this project, we are asked to identify abnormal behaviors in a system, which relies on the analysis of logs collected in real-time from the log aggregation systems of an enterprise.
This is server log format
```
IP [TIME] [Method Path] StatusCode ResponseLength [[UserAgent]] ResponseTime
```
and this is a generated sample
```
42.236.10.125 [2020-12-19T15:23:10.0+0100] [GET / http://baidu.com/] 200 10479 [["Mozilla/5.0 (Linux; U; Android 8.1.0; zh-CN; EML-AL00 Build/HUAWEIEML-AL00) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/57.0.2987.108]] 32
```
### Preprocessing
Data preprocessing has two main parts, feature extraction and feature transformation.
#### Feature Extraction
`tehran_traffic_statistics`: Based on [Tehran-IX](http://members.tehran-ix.ir/statistics/ixp/pkts) Packets/Second in different hours of daytime, gives a weight to each request. It is highly crawler in hours like 6 a.m.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/tehran-ix.png)

`url_depth`:  Depth of the request URL.
`is_spider`: Extracted from user agent, specify that request is from a bot.
`is_phone`: Extracted from user agent, specify that request is from a phone or a PC.
#### Feature Transformation
In feature transformation we used three different methods, bucketing, normalization and one-hot encoding.
`response_length`: Bucketing on a log scale using `np.geomspace()` into `['zero', 'small', 'medium', 'big']` scales.
`requested_file_types`: Bucketing into `['img', 'code', 'renderable', 'app', 'video', 'font', 'endpoint']` data types.
`status_code`: Bucketing into `['is_1xx', 'is_2xx', 'is_3xx', 'is_4xx', 'is_5xx']` status codes.
`method`: One-hot encoding request methods.
`time_weight`: Normalized extracted `tehran_traffic_statistics`.
### Learning Algorithm
#### PCA
PCA is an unsupervised machine learning algorithm that attempts to reduce the dimensionality. Using PCA, you can reduce the dimensionality data and reconstruct the it. Since anomaly show the largest reconstruction error, abnormalities can be found based on the error between the original data and the reconstructed data. [Here](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/reconstructed_sample.png) is a reconstructed sample and calculated error.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/pca_evaluation.png)

#### Isolation Forest
Isolation forest works on the principle of the decision tree algorithm. Due to the fact that anomalies require fewer random partitions than normal normal data points in the data set. So anomalies are points that have a shorter path in the tree.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/isolation_forest_evaluation.png)

#### Autoencoder
Let's get deeper. An autoencoder is a special type of neural network that copies the input values to the output values. It does not require the target variable like the conventional Y, thus it is categorized as unsupervised learning.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/autoencoder_evaluation.png)

We also plot ROC curve for models to select the best model and an appropriate threshold.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/ROC.png)

We chose PCA for the final model, but can we follow some rules to get a better model? Let's use hybrid approach.
#### Hybrid Approach
In hybrid approach we combined PCA with rule based model. We added two rules, requests that says they are spiders and requests that use known malicious IPs. Evaluation shows that the model is good enough.

![](https://github.com/zivdar001matin/web-crawler-detection/blob/main/icons/hybrid_evaluation.png)

### API
API was implemented using MLflow. MLflow is an open source platform to manage the ML lifecycle.
You can see API example in [this video](https://drive.google.com/file/d/1IzT8EME9zJyGkGMINLrx-F6jT8BfwVuM/view?usp=sharing). Sent requests using Postmman.
## Contact
Our Team:
- Ahmad Etezadi - etezadi63@gmail.com
- Matin Zivdar - zivdar1matin@gmail.com

Special thanks to our supportive and knowledgeable mentor  Tadeh Alexani - [@tadeha](https://github.com/tadeha) and Rahnema College Team.
