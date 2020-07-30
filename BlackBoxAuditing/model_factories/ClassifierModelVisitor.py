import numpy as np
from BlackBoxAuditing.model_factories.AbstractModelVisitor import AbstractModelVisitor

class ClassifierModelVisitor(AbstractModelVisitor):
    def __init__(self, trained_classifier, predict_function, label_index = 0):

        """
        Creating an instance of ClassifierModelVisitor class lets a user use thier own pre-trained 
        classifier model (that they want to audit). Note that the classfier needs to have a predict 
        function that predicts the class labels for the provided data.

        Parameters:
        ----------------
        trained_classifier : a pre-trained model object
            This is the model you want to audit. The object needs to have a predict method 
            that predicts the class labels for the provided data.

        predict_function : a method for trained_classifier
            A function that predicts the class labels for the provided test set.
            In the case of sklearn model, this input should be trained_classifier.predict 

        label_index : int
            Indicates which column corresponds to the label. The default is the first column.
        """

        super(ClassifierModelVisitor,self).__init__("Pretrained")
        self.classifier = trained_classifier
        self.predict_function = predict_function
        self.label_index = label_index

    def get_Xy(self, test_set):

        """
        This is a helper function to obtain X (test data) and y (test label) for a test set.

        Parameters:
        ----------------
        test_set : list of list (matrix) or np.array with the dimension of (# of samples)*(# of features + 1).
            We use this test_set to audit the model.

        Returns:
        ----------------
        X, y : np.matrix, np.array
            X indicates test data and y indicates test label for the test set.

        """
        X = np.matrix([row[:self.label_index] + row[self.label_index+1:] for row in test_set])
        y = np.asarray([row[self.label_index] for row in test_set])
        return X, y

    def test(self, test_set, test_name = ""):
        
        """
        This function checks the accuracy of model's predictions compared with the real label.
        If the data combines the data and the labels, we can use get_Xy function above.

        Parameters:
        ----------------
        test_set : list of list (matrix) or np.array with the dimension of (# of samples)*(# of features + 1).
            Test dataset for auditing.

        test_name : str
            The name of tests

        Returns:
        ----------------
        a list of tuples
            Each tuple are consisted of the true label and the prediction by the model.

        """            
        X, y = self.get_Xy(test_set)
        predictions = self.predict_function(X)
        return list(zip(y, predictions))

def test():
    test_basic_model()

def test_basic_model():
    mock = MockModel()
    model = ClassifierModelVisitor(mock, mock.predict, 0)
    a = [[0,2],[0,2]]
    output = model.test(a)
    output_list = [ x for x in output ]
    correct = zip([0, 0], [1, 1])
    correct_list = [ x for x in correct ]
    print("correct basic pre-trained model? -- ", output_list == correct_list)

class MockModel():
    def predict(self, X):
        return [1 for x in X]
        
if __name__=='__main__':
    test()
