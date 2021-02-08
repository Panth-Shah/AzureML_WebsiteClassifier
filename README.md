# Microsoft Azure Captsone Project: Detect Malicious Websites

## Project Abstract: 

Over the past few years, Web Security has become a growing concern for internet users and organizations asthey rely on multiple web-sites for their daily tasks like shopping, banking, information retrival etc. With growing use of internet, number of malicious websites have also grown exponentially, developed by attackers with an intention to breach individual privacy and steal data to use it for fraudulant activities. This act of creating fake websites, which are in many cases imitating real organizations is a serious concern for many due to increasing number of scams using stolen identity and data theft. Internet users and many organizations have been a victim of phishing and other internat crimes, simply because no accurate classificaton can be obrained between malicious and benign websites simply by viewing content of the website.

A log term-goal of this project will be to develop a real-time system that uses Machine Learning techniques to detect malicious URLs (spam, phishing, exploits, etc.). Techniques explored involving classifying URLs based on their **Lexical** and **Host-based** features, as well as online learning to process large numbers of examples and adapt quickly to evolving URLs over time are captured from research published by PhD scholars from UC San Diago [1] & [2]. This project aims to extend research finding and construct a classifier which can predict malicious and benign website URLs, based on application layer and network characteristics utilizing captured data for this research work.

## Dataset:

### Dataset Overview:

For this capstone project, we have used [Malicious and Benign Websites](https://www.kaggle.com/xwolf12/malicious-and-benign-websites) dataset. Dataset is created with data obtained from verified sources of benign and malicious URL's in a low interactive client honeypot to isolate network traffic. To study malicious website, application and network layer features are identified, which are listed under file strucutre. There are 1781 unique URL records with 21 different featurs. Here, we will not include Web page content or the context of the URL as a feature to avoid downloading content as classifying URL with a trained odel is a lightweight operation compared to first downloading the page content and then analyzing them.  Appraoch with this dataset is to classify URLs without looking at the content of the website as malicious site may serve benign versions of a page to honeypot IP address run by a security practitioner, but serve alicious version to other user due to content "cloaking". Type of features are as follows:

1. **Lexical Features:** These features allow us to capture the property that malicious URLs tend to hold different from benign URLs, for example URL domain names, keywords and lengths of the hostname (phish.biz/www.indeed.com/index.php or www.craigslist.com.phishy.biz)

2. **Host-based features:** These features allow us to identify where malicious website is hosted from, who owns them, and how they are managed.

	- **WHOIS Information:** This includes domain name registration dates, registrars, and registrants. Using this feature, we can tag all the websites as malicious registered by the same individual and such ownership as a malicious feature.

	- **Location:** his refers to the host's geo-location, IP Address prefix, and autonomous system (AS) number. So websites hosted in a specific IP prefix of an Internet Service Provider can be tagged as a malicious website and account for such host can be classified disreputable ISP when classifying URLs.
	
	- **Connection Speed:** Connection speed of some malicious websites residing on compormised residential machines.
	
	- **Other DNS related properties:** These include time-to-live (TTL), spam-related domain name heuristics, and whether the DNS records share the same ISP. 

Below are the resources used to create this dataset.

- machinelearning.inginf.units.it/data-andtools/hidden-fraudulent-urls-dataset
- malwaredomainlist.com
- https://github.com/mitchellkrogza/The-Big-List-of-Hacked-Malware-Web-Sites

### File Structure:

Column Names | Details
------------ | -------------
`URL` | It is the anonimous identification of the URL analyzed in the study.
`URL_LENGTH` | It is the number of characters in the URL.
`NUMBER_SPECIAL_CHARACTERS` | It is number of special characters identified in the URL, such as, “/”, “%”, “#”, “&”, “. “, “=”.
`CHARSET` | It is a categorical value and its meaning is the character encoding standard (also called character set).
`SERVER` | It is a categorical value and its meaning is the operative system of the server got from the packet response.
`CONTENT_LENGTH` | It represents the content size of the HTTP header.
`WHOIS_COUNTRY` | It is a categorical variable, its values are the countries we got from the server response (specifically, our script used the API of Whois).
`WHOIS_STATEPRO` | It is a categorical variable, its values are the states we got from the server response (specifically, our script used the API of Whois).
`WHOIS_REGDATE` | Whois provides the server registration date, so, this variable has date values with format DD/MM/YYY HH:MM
`WHOIS_UPDATEDDATE` | Through the Whois we got the last update date from the server analyzed
`TCP_CONVERSATION_EXCHANGE` | This variable is the number of TCP packets exchanged between the server and our honeypot client
`DIST_REMOTETCP_PORT` | It is the number of the ports detected and different to TCP
`REMOTE_IPS` | This variable has the total number of IPs connected to the honeypot
`APP_BYTES` | This is the number of bytes transfered
`SOURCE_APP_PACKETS` | Packets sent from the honeypot to the server
`REMOTE_APP_PACKETS` | Packets received from the server
`APP_PACKETS` | This is the total number of IP packets generated during the communication between the honeypot and the server
`DNS_QUERY_TIMES` | This is the number of DNS packets generated during the communication between the honeypot and the server
`TYPE` | This is a categorical variable, its values represent the type of web page analyzed, specifically, 1 is for malicious websites and 0 is for benign websites

### Task:

In this capstone project, we aim to create a model to classify if a website URL is `Malicious` or `Benign` with the use of dataset features explaned above. To achieve this, we will be using two approaches and compare both using `Accuracy` as a Primary Metric.

1. **Using [AutomatedML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)** With this approach, we will be using Microsoft Azure's Automated ML feature to train and tune a model for given dataset to predict which category (Maclicious or Benign) new URL will fall into based on learnings from it's training data. In this approach, Azure Machine Learning taking user inputs such as `Dataset`, `Target Metric` and `Constraints` into account, train model into multiple iterations and will return best performing model with highest training score achieved.

2. **Using [HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py):** With this approach, we will train a Scikit-learn Logistic Regression model and automating hyperparaeter tuning by using Azure ML's Hyperdrive package. By defining hyperparameter space, we will tune model applying different combinations of hyperparameters and tuning it untill we find the best performing model. Here, models will be compared on primary metrics defined. Unlike AutoML, with this approach we will need to manually perform feature scaling, normalization and other data preprocessing on our dataset to reduce overfitting and effect of bad data on model performance.

Post model training using both the approches, we will be comparing performance of both the models using performance metrics **Accuracy** and best performing model will be deployed on **Azure Container Instance (ACI)** as a web service registered with Azure workspace and open to consume by external services with provided autorization. Finally, functionality of the deployed model will be demonstrated using response received for each successful HTTP POST Request to an end-point for real-time inferencing.

### Access:

Azure mainly supports two types of Datasets: **A. FileDataset B. TabularDataset**. Here, we have data captured in **csv file**, which can be handled using **TabularDataset** as it is used for tabular data. Dataset is uploaded to [github repository](https://github.com/Panth-Shah/nd00333-capstone/blob/master/Dataset/malicious_website_dataset.csv), which is later used to register datastore with Azure ML Workspace using `Dataset.Tabular.from_delimited_files()`. We can also creare a new TabularDataset by directly calling the corresponding factory methods of the class defined in `TabularDatasetFactory`.

	# Create AML Dataset and register it into Workspace
	example_data = 'https://raw.githubusercontent.com/Panth-Shah/nd00333-capstone/master/Dataset/malicious_website_dataset.csv'
	dataset = Dataset.Tabular.from_delimited_files(example_data)

	# Create TabularDataset using TabularDatasetFactory
	dataset = TabularDatasetFactory.from_delimited_files(path=example_data)

	#Register Dataset in Workspace
	dataset = dataset.register(workspace=ws, name=key, description=description_text)

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
