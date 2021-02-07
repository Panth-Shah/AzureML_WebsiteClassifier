# Microsoft Azure Captsone Project: Detect Malicious Websites

## Project Abstract: 

Over the past few years, Web Security has become a growing concern for internet users and organizations asthey rely on multiple web-sites for their daily tasks like shopping, banking, information retrival etc. With growing use of internet, number of malicious websites have also grown exponentially, developed by attackers with an intention to breach individual privacy and steal data to use it for fraudulant activities. This act of creating fake websites, which are in many cases imitating real organizations is a serious concern for many due to increasing number of scams using stolen identity and data theft. Internet users and many organizations have been a victim of phishing and other internat crimes, simply because no accurate classificaton can be obrained between malicious and benign websites simply by viewing content of the website.

A log term-goal of this project will be to develop a real-time system that uses Machine Learning techniques to detect malicious URLs (spam, phishing, exploits, etc.). Techniques explored involving classifying URLs based on their lexical and host-based features, as well as online learning to process large numbers of examples and adapt quickly to evolving URLs over time are captured from research published by PhD scholars from UC San Diago [1] & [2]. This project aims to extend research finding and construct a classifier which can predict malicious and benign website URLs, based on application layer and network characteristics utilizing captured data for this research work.

## Project Scope:

This capstone project is a part of Udacity's Azure Machine Learning Engineer Nanodegree sponsored by Microsoft. Objective of this project is to construct a Machine Learning model using an external dataset not available inside Azure ecosystem using two different techniques: A. Using [AutomatedML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) B. Constomized model by tuning hyperparameters using [HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py). Post model training using both the approches, we will be comparing performance of both the models using performance metrics and best performing model will be deployed on **Azure Container Instance (ACI)** as a web service registered with Azure workspace and open to consume by external services with provided autorization. Finally, functionality of the deployed model will be demonstrated using response received for each successful HTTP POST Request to an end-point for real-time inferencing.

## Dataset Information:

For this capstone project, we have used [Malicious and Benign Websites](https://www.kaggle.com/xwolf12/malicious-and-benign-websites) dataset. Dataset is created with data obtained from verified sources of benign and malicious URL's in a low interactive client honeypot to isolate network traffic. To study malicious website, application and network layer features are identified, which are listed under file strucutre. Below are the resources used to create this dataset.

	- machinelearning.inginf.units.it/data-andtools/hidden-fraudulent-urls-dataset
	- malwaredomainlist.com
	- https://github.com/mitchellkrogza/The-Big-List-of-Hacked-Malware-Web-Sites

### File Structure:


Column Names | Details
------------ | -------------
`URL` | It is the anonimous identification of the URL analyzed in the study.
`URL_LENGTH` | It is the number of characters in the URL.
`NUMBERSPECIALCHARACTERS` | It is number of special characters identified in the URL, such as, “/”, “%”, “#”, “&”, “. “, “=”.
`CHARSET` | It is a categorical value and its meaning is the character encoding standard (also called character set).
`SERVER` | It is a categorical value and its meaning is the operative system of the server got from the packet response.
`CONTENT_LENGTH` | it represents the content size of the HTTP header.
`WHOIS_COUNTRY` | it is a categorical variable, its values are the countries we got from the server response (specifically, our script used the API of Whois).
`WHOIS_STATEPRO` | it is a categorical variable, its values are the states we got from the server response (specifically, our script used the API of Whois).
`WHOIS_REGDATE` | Whois provides the server registration date, so, this variable has date values with format DD/MM/YYY HH:MM
`WHOISUPDATEDDATE` | Through the Whois we got the last update date from the server analyzed
`TCPCONVERSATIONEXCHANGE` | This variable is the number of TCP packets exchanged between the server and our honeypot client
`DISTREMOTETCP_PORT` | it is the number of the ports detected and different to TCP
`REMOTE_IPS` | this variable has the total number of IPs connected to the honeypot
`APP_BYTES` | this is the number of bytes transfered
`SOURCEAPPPACKETS` | packets sent from the honeypot to the server
`REMOTEAPPPACKETS` | packets received from the server
`APP_PACKETS` | this is the total number of IP packets generated during the communication between the honeypot and the server
`DNSQUERYTIMES` | this is the number of DNS packets generated during the communication between the honeypot and the server
`TYPE` | this is a categorical variable, its values represent the type of web page analyzed, specifically, 1 is for malicious websites and 0 is for benign websites

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

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
