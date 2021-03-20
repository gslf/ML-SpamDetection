# :#/ GSLF
# Gioele SL Fierro
# 2021
 
import re, email, time

import pickle

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class SpamDetector:
    '''
    An effective tool for SPAM classification that use Nive Bayes classification, a simple Machine Learning technique.

    ...

    Attributes
    ----------
    dataset_path : str
        Path of the "trec07" dataset
    labels_path : str
        Path of the labels for the selected dataset
    train_test_ratio : float
        Ratio betweet train set and test set (Default value - 0.7).

    Methods
    -------
    startTraining():
        Train the classifier.

    loadWeights(weights_path):
        Load weights from previous training.

    classify(mail_path):
        Classify an email

    '''

    def __init__(self, dataset_path, labels_path, train_test_ratio = 0.7):
        '''
        Constructor
        '''

        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.train_test_ratio = train_test_ratio

        self.mails = []
        self.labels = []
        self.classifier = None
        self.vectorizer = None

    def _loadDataset(self):
        # Labels to dictionary
        labels_dictionary = {}
        with open(self.labels_path) as label_file:
            for line in label_file:
                    label_id = re.findall(r'\d+', line)[0]

                    if "spam" in line:
                        labels_dictionary[label_id] = False
                    else:
                        labels_dictionary[label_id] = True

        # Load Email + Label
        for label_id in labels_dictionary.keys():
            file_path = self.dataset_path + 'inmail.' + label_id
            mail_content = self._read_mail(file_path)

            self.mails.append(mail_content)
            self.labels.append(labels_dictionary[label_id])

    def _read_mail(self, mail_path):
        mail_content = ''

        with open(mail_path, encoding='utf-8', errors='replace') as mail_file:

            mail_message = email.message_from_file(mail_file)

            # Read subject
            if mail_message and mail_message['Subject']:
                mail_content += mail_message['Subject']
                mail_content += ' '

            # Read body
            if mail_message and mail_message['Subject']:
                mail_content += mail_message['Subject']
                mail_content += ' '
            
            mail_content += self._extract_body(mail_message)

        return mail_content

    def _extract_body(self, mail_message):
        body = ''
        for part in mail_message.walk():
            if part.get_content_type() == 'text/plain' or part.get_content_type() == 'text/html':
                body += str(part.get_payload())

        return body

    def startTraining(self, weights_path = "weights.h5"):
        '''
        Train a classifier and save weights to file

        Parameters
        ----------
        weights_path : str, optional
            Path of the weights file (default is "weights.h5")

        Returns
        -------
        result : boolean
            Result of training session
        '''

        self._loadDataset()
        mail_train, mail_test, labels_train, labels_test, _ , _ = train_test_split(
            self.mails, 
            self.labels, 
            range(len(self.labels)), 
            train_size = self.train_test_ratio, 
            random_state = 2)

        # Vectorization
        self.vectorizer = CountVectorizer()
        mail_train_vector = self.vectorizer.fit_transform(mail_train)
        mail_test_vector = self.vectorizer.transform(mail_test)

        # Naive Bayes Classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(mail_train_vector, labels_train)

        # Save model
        try:
            pickle.dump(self.classifier, open(weights_path, 'wb'))
            pickle.dump(self.vectorizer, open(weights_path + '.vec', 'wb'))
        except Exception as exptn:
            print("WARNING: Isn't possible to save the model on disk!")
        

        # Measurements
        predictions = self.classifier.predict(mail_test_vector)
        training_result = classification_report(labels_test, predictions, target_names=['SPAM', 'NOT SPAM'])
        training_result += '\n' + 'Classification accuracy {:.1%}'.format(accuracy_score(labels_test, predictions))
        return training_result


    def loadWeights(self, weights_path  = "weights.h5"):
        '''
        Load a pretrained classifier

        Parameters
        ----------
        weights_path : str, optional
            Path of the weights file (default is "weights.h5")

        Returns
        -------
        result : boolean
            Result of file loading
        '''
        try:
            self.classifier = pickle.load(open(weights_path, 'rb'))
            self.vectorizer = pickle.load(open(weights_path + '.vec', 'rb'))
            return True
        except Exception as exptn:
            return False

    def classify(self, mail_path):
        '''
        Classify an email from file

        Parameters
        ----------
        mail_path : str, required
            The path of the mail message

        Returns
        -------
        result : string
            Result of classification
                "SPAM" > The mail is spam
                "HAM" > The mail isn't spam
                "ERROR: . . . " > An error as occourred
        '''
        if self.classifier != None and self.vectorizer != None:
            # Read mail
            mail = [self._read_mail(mail_path)]

            # Vectorize mail
            mail_vector = self.vectorizer.transform(mail)

            # Classification
            prediction = self.classifier.predict(mail_vector)

            if prediction:
                return "NOT SPAM"
            else:
                return "SPAM"

        else:
            return "ERROR: No trained model. Start a training or load pretrained weights."
        
    

