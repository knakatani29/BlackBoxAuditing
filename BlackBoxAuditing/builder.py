import csv
import os
import pandas

from datetime import datetime

from BlackBoxAuditing.model_factories import SVM, DecisionTree, NeuralNetwork
from BlackBoxAuditing.model_factories.SKLearnModelVisitor import SKLearnModelVisitor
from BlackBoxAuditing.loggers import vprint
from BlackBoxAuditing.GradientFeatureAuditor import GradientFeatureAuditor
from BlackBoxAuditing.audit_reading import graph_audit, graph_audits, rank_audit_files, group_audit_ranks
from BlackBoxAuditing.consistency_graph import graph_prediction_consistency
from BlackBoxAuditing.measurements import get_conf_matrix, accuracy, BCR
from BlackBoxAuditing.data import load_data, load_from_file, load_testdf_only

"""
The builder class builds a model using the data given.
"""

class Builder():
  def __init__(self, model_options = {}, verbose = True, RETRAIN_MODEL_PER_REPAIR= False, ModelFactory = SVM, kdd = False, _audits_data = {}):
  
    self.model_options = model_options
    self.verbose = verbose
    self.RETRAIN_MODEL_PER_REPAIR = RETRAIN_MODEL_PER_REPAIR
    self.ModelFactory = ModelFactory
    self.kdd = kdd
    self._audits_data = _audits_data

  def __call__(self, data, output_dir=None, dump_all=False, features_to_audit=None):
    start_time = datetime.now()

    headers, train_set, test_set, response_header, features_to_ignore, correct_types = data

    self._audits_data = {"headers" : headers, "train" : train_set, "test" : test_set,
                         "response" : response_header, "ignore" : features_to_ignore,
                         "types" : correct_types,
                         "full_audit" : True if features_to_audit is None else False
                        }

    
      """
       ModelFactories require a `build` method that accepts some training data
       with which to train a brand new model. This `build` method should output
       a Model object that has a `test` method -- which, when given test data
       in the same format as the training data, yields a confusion table detailing
       the correct and incorrect predictions of the model.
      """

      all_data = train_set + test_set
      model_factory = self.ModelFactory(all_data, headers, response_header,
                                        features_to_ignore=features_to_ignore,
                                        options=self.model_options)

    if self.trained_model != None:
      model_or_factory = self.trained_model
    elif not self.RETRAIN_MODEL_PER_REPAIR:
      vprint("Training initial model.",self.verbose)
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

      model_or_factory = model
      
      return model
    else:
      model_or_factory = model_factory
