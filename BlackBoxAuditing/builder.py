import csv
import os

from BlackBoxAuditing.model_factories import SVM, DecisionTree, NeuralNetwork
from BlackBoxAuditing.loggers import vprint
from BlackBoxAuditing.measurements import get_conf_matrix, accuracy, BCR
from BlackBoxAuditing.data import load_data, load_from_file, load_testdf_only


class Builder():
  def __init__(self, measurers = [accuracy, BCR], model_options = {}, 
                verbose = True, ModelFactory = SVM):
  
    """
    Uses this Builder class to train a model using model factories.
    ModelFactories require a `build` method that accepts some training data
    with which to train a brand new model. This `build` method should output
    a Model object that has a `test` method -- which, when given test data
    in the same format as the training data, yields a confusion table detailing
    the correct and incorrect predictions of the model.
    
    Parameters
    ----------

    measurers : list of measurer (default : [accuracy, BCR])
        List of measurers to use for Gradient Feature Auditing.
        Check measurements.py for available measurements (accuracy and BCR as of 2020/07/28).

    model_options : dictionary (default : {})
        Model options needed to set parameters for training a model in model factory. 

    verbose : boolean value (default : True)
        Allows more detailed status updates while auditing.

    modelfactory : ModelFactory Class (default : SVM)
        When we don't have a pretrained model (modelvisitor), we use the method indicated here to train 
        a model. We can either use SVM, DecisionTree, NeuralNetwork, or we can create a new class.

    """

    self.measurers = measurers
    self.model_options = model_options
    self.verbose = verbose
    self.ModelFactory = ModelFactory

  def train(self, train_set, test_set, headers, response_header, features_to_ignore = []):

    """
    A method to train a model using model factories. 
    ModelFactories require a `build` method that accepts some training data
    with which to train a brand new model. This `build` method should output
    a Model object that has a `test` method -- which, when given test data
    in the same format as the training data, yields a confusion table detailing
    the correct and incorrect predictions of the model.
    
    Parameters
    ----------
    train_set, test_set : list of list or numpy.array with teh dimensions (# of features)*(# of samples).
      Data for training the model and testing the model.

    headers : list of strings
      The headers of the data.

    response_header : string
      The response header of the data.

    features_to_ignore : list of strings (default = [])
      The features we want to ignore.

    """

    all_data = train_set + test_set
    model_factory = self.ModelFactory(all_data, headers, response_header,
                                      features_to_ignore=features_to_ignore,
                                      options=self.model_options)

    vprint("Training initial model.", self.verbose)
    model = model_factory.build(train_set)

    # Check the quality of the initial model on verbose runs.
    if self.verbose:
      print("Calculating original model statistics on test data:")
      print("\tTraining Set:")
      train_pred_tuples = model.test(train_set)
      train_conf_matrix = get_conf_matrix(train_pred_tuples)
      print("\t\tConf-Matrix:", train_conf_matrix)
      for measurer in self.measurers:
        print("\t\t{}: {}".format(measurer.__name__, measurer(train_conf_matrix)))

      print("\tTesting Set:")
      test_pred_tuples = model.test(test_set)
      test_conf_matrix = get_conf_matrix(test_pred_tuples)
      print("\t\tConf-Matrix", test_conf_matrix)
      for measurer in self.measurers:
        print("\t\t{}: {}".format(measurer.__name__, measurer(test_conf_matrix)))

    return model

def test():
  pass