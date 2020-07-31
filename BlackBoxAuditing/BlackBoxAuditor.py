import csv
import os
import pandas

from datetime import datetime

from BlackBoxAuditing.model_factories import SVM, DecisionTree, NeuralNetwork, ClassifierModelVisitor
from BlackBoxAuditing.loggers import vprint
from BlackBoxAuditing.GradientFeatureAuditor import GradientFeatureAuditor
from BlackBoxAuditing.audit_reading import graph_audit, graph_audits, rank_audit_files, group_audit_ranks
from BlackBoxAuditing.consistency_graph import graph_prediction_consistency
from BlackBoxAuditing.measurements import get_conf_matrix, accuracy, BCR
from BlackBoxAuditing.data import load_data, load_from_file, load_testdf_only
from builder import Builder


class Auditor():
  def __init__(self, measurers = [accuracy, BCR], model_options = {}, verbose = True, REPAIR_STEPS = 10,
                RETRAIN_MODEL_PER_REPAIR= False, WRITE_ORIGINAL_PREDICTIONS = True, modelfactory = SVM, 
                modelvisitor = None, kdd = False):

    """
    Uses this Auditor class to audit a model. After initializing with this initializer,
    we can use __call__ method to audit a model. 
    
    Parameters
    ----------

    measurers : list of measurer (default : [accuracy, BCR])
        List of measurers to use for Gradient Feature Auditing.
        Check measurements.py for available measurements (accuracy and BCR as of 2020/07/28).

    model_options : dictionary (default : {})
        Model options needed to set parameters for training a model in model factory. 

    verbose : boolean value (default : True)
        Allows more detailed status updates while auditing.

    REPAIR_STEPS : int (default = 10) 
        Number of repair steps to take. 
    
    RETRAIN_MODEL_PER_REPAIR : boolen value (default = False)
        Indicates if a model needs to be re-trained per repair. If false, the model will be trained once
        before GFA (gradient feature auditing), and then won't be trained again in the process.

    WRITE_ORIGINAL_PREDICTIONS : boolean value (default = True)
        Allows writing out the original prediction.

    modelfactory : ModelFactory Class (default : SVM)
        When we don't have a pretrained model (modelvisitor), we use the method indicated here to train 
        a model. We can either use SVM, DecisionTree, NeuralNetwork, or we can create a new class.

    modelvisitor : ModelVisitor instance  (default : None)
        Allows the use of a pretrained model. We can create an instance of ClassifierModelVisitor 
        (refer to model_factories/ClassifierModelVisitor.py) and assign it as modelvisitor to use it.
        When we have a modelvisitor, we don't use the method indicated in modelfactory to train a model.

    kdd : boolean value (default : False)

    """
    self.measurers = measurers
    self.model_options = model_options
    self.verbose = verbose
    self.REPAIR_STEPS = REPAIR_STEPS
    self.RETRAIN_MODEL_PER_REPAIR = RETRAIN_MODEL_PER_REPAIR
    self.WRITE_ORIGINAL_PREDICTIONS = WRITE_ORIGINAL_PREDICTIONS
    self.modelfactory = modelfactory
    self.modelvisitor = modelvisitor
    self.kdd = kdd
    self._audits_data = {}
    self.trained = False
	
  def __call__(self, data, output_dir=None, dump_all=False, features_to_audit=None):
    start_time = datetime.now()

    headers, train_set, test_set, response_header, features_to_ignore, correct_types = data

    self._audits_data = {"headers" : headers, "train" : train_set, "test" : test_set,
                         "response" : response_header, "ignore" : features_to_ignore,
                         "types" : correct_types,
                         "full_audit" : True if features_to_audit is None else False
                        }

    if self.modelvisitor != None:
      # Passing the pretrained model to GFA.
      self.trained = True
      model_to_audit = self.modelvisitor

    elif not self.RETRAIN_MODEL_PER_REPAIR:
      # Initial train since we are not retraining the model per repair.
      model = Builder(measurers = self.measurers, model_options = self.model_options, 
                verbose = self.verbose, ModelFactory = self.modelfactory)
      model_to_audit = model.train(train_set, test_set, headers, response_header, features_to_ignore=features_to_ignore)
      self.trained = True

    else:
      # We retrain the model for each repair process.
      model_to_audit = self.modelfactory

    # Translate the headers into indexes for the auditor.
    audit_indices_to_ignore = [headers.index(f) for f in features_to_ignore]

    # Don't audit the response feature.
    audit_indices_to_ignore.append(headers.index(response_header))

    # Prepare the auditor.
    auditor = GradientFeatureAuditor(model_to_audit, headers, train_set, test_set,
                                     repair_steps=self.REPAIR_STEPS, kdd=self.kdd,
                                     features_to_ignore=audit_indices_to_ignore,
                                     features_to_audit=features_to_audit,
                                     output_dir=output_dir,dump_all=dump_all)

    # Perform the Gradient Feature Audit and dump the audit results into files.
    audit_filenames = auditor.audit(verbose=self.verbose)

    # Retrieve repaired data from audit
    self._audits_data["rep_test"] = auditor._rep_test

    ranked_features = []
    for measurer in self.measurers:
      vprint("Ranking audit files by {}.".format(measurer.__name__),self.verbose)
      #ranked_graph_filename = "{}/{}.png".format(auditor.OUTPUT_DIR, measurer.__name__)
      ranks = rank_audit_files(audit_filenames, measurer)
      vprint("\t{}".format(ranks), self.verbose)
      ranked_features.append( (measurer, ranks) )

    end_time = datetime.now()

    # Store a summary of this experiment.
    model_id = self.modelfactory.factory_name if self.modelvisitor == None else "Pretrained"
    model_name = self.modelfactory.verbose_factory_name if self.modelvisitor == None else "Pretrained"
    summary = [
      "Audit Start Time: {}".format(start_time),
      "Audit End Time: {}".format(end_time),
      "Retrained Per Repair: {}".format(self.RETRAIN_MODEL_PER_REPAIR),
      "Model Factory ID: {}".format(model_id),
      "Model Type: {}".format(model_name),
      "Non-standard Model Options: {}".format(self.model_options),
      "Train Size: {}".format(len(train_set)),
      "Test Size: {}".format(len(test_set)),
      "Non-standard Ignored Features: {}".format(features_to_ignore),
      "Features: {}\n".format(headers)]

    # Print summary
    for line in summary:
      print(line)

    for ranker, ranks in ranked_features:
      print("Ranked Features by {}: {}".format(ranker.__name__, ranks))
      groups = group_audit_ranks(audit_filenames, ranker)
      print("\tApprox. Trend Groups: {}\n".format(groups))

      if ranker.__name__ == "accuracy":
        self._audits_data["ranks"] = ranks

    if self.trained:
      train_pred_tuples = model_to_audit.test(train_set)
      test_pred_tuples = model_to_audit.test(test_set)
    else:
      if self.WRITE_ORIGINAL_PREDICTIONS:
        raise ValueError("The model is not pretrained. There is no original prediction.")

    # Dump all experiment results if opted
    if dump_all:
      vprint("Dumping original training data.", self.verbose)
      # Dump the train data to the log.
      train_dump = "{}/original_train_data".format(auditor.OUTPUT_DIR)
      with open(train_dump + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in train_set:
          writer.writerow(row)

      if self.WRITE_ORIGINAL_PREDICTIONS:
        # Dump the predictions on the test data.
        with open(train_dump + ".predictions", "w") as f:
          writer = csv.writer(f)
          file_headers = ["Response", "Prediction"]
          writer.writerow(file_headers)
          for response, guess in train_pred_tuples:
            writer.writerow([response, guess])

      vprint("Dumping original testing data.", self.verbose)
      # Dump the train data to the log.
      test_dump = "{}/original_test_data".format(auditor.OUTPUT_DIR)
      with open(test_dump + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in test_set:
          writer.writerow(row)

      if self.WRITE_ORIGINAL_PREDICTIONS:
        # Dump the predictions on the test data.
        with open(test_dump + ".predictions", "w") as f:
          writer = csv.writer(f)
          file_headers = ["Response", "Prediction"]
          writer.writerow(file_headers)
          for response, guess in test_pred_tuples:
            writer.writerow([response, guess])

      # Graph the audit files.
      vprint("Graphing audit files.",self.verbose)
      for audit_filename in audit_filenames:
        audit_image_filename = audit_filename + ".png"
        graph_audit(audit_filename, self.measurers, audit_image_filename)

      # Store a graph of how many predictions change as features are repaired.
      vprint("Graphing prediction changes throughout repair.",self.verbose)
      output_image = auditor.OUTPUT_DIR + "/similarity_to_original_predictions.png"
      graph_prediction_consistency(auditor.OUTPUT_DIR, output_image)

    for measurer in self.measurers:
       ranked_graph_filename = "{}/{}.png".format(auditor.OUTPUT_DIR, measurer.__name__)
       graph_audits(audit_filenames, measurer, ranked_graph_filename)

    # Store a summary of this experiment to file.
    summary_file = "{}/summary.txt".format(auditor.OUTPUT_DIR)
    with open(summary_file, "w") as f:
      for line in summary:
        f.write(line+'\n')

      for ranker, ranks in ranked_features:
        f.write("Ranked Features by {}: {}\n".format(ranker.__name__, ranks))
        groups = group_audit_ranks(audit_filenames, ranker)
        f.write("\tApprox. Trend Groups: {}\n".format(groups))

    vprint("Summary file written to: {}\n".format(summary_file), self.verbose)

  def find_contexts(self, removed_attr, output_dir, beam_width=10, min_covered_examples=1, max_rule_length=5, by_original=True, epsilon=0.05):
    # import done here so that Orange install is optional
    from BlackBoxAuditing.find_contexts import context_finder, load

    # retrive data from the audit
    audits_data = self._audits_data
    full_audit = audits_data["full_audit"]
    # Make sure a full audit was completed
    if not full_audit:
      raise RuntimeError("Only partial audit completed. Must run a full audit to call find_contexts.")

    orig_train = audits_data["train"]
    orig_test = audits_data["test"]
    obscured_test_data = audits_data["rep_test"][removed_attr]
    headers = audits_data["headers"]
    response_header = audits_data["response"]
    features_to_ignore = audits_data["ignore"]
    correct_types = audits_data["types"]
    obscured_tag = "-no"+removed_attr

    # Create directory to dump results
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Extract influence scores
    ranks = audits_data["ranks"]
    influence_scores = {}
    for element in ranks:
      influence_scores[element[0]] = float(element[1])
      influence_scores[element[0]+obscured_tag] = 0.0

    # Get obscured data from file:
    obscured_test = []
    obscured_test_reader = csv.reader(open(obscured_test_data, 'r'))
    for row in obscured_test_reader:
      obscured_test.append(row)

    # load data from audit to prepare it for context finding process
    audit_params = (orig_train, orig_test, obscured_test, headers, response_header, features_to_ignore, correct_types, obscured_tag)

    orig_train_tab, orig_test_tab, merged_data = load(audit_params, output_dir)

    # run the context_finder
    context_finder(orig_train, orig_test, obscured_test, orig_train_tab, orig_test_tab, merged_data, obscured_tag, output_dir, influence_scores, beam_width, min_covered_examples, max_rule_length, by_original, epsilon)

### Tests and examples below

def german_example_audit():
  # format data
  data = load_data("german")

  # set the auditor
  auditor = Auditor()
  auditor.model = SVM

  # call the auditor
  auditor(data, output_dir="german_audit_output", dump_all=False)

  auditor.find_contexts("age_cat",output_dir="german_context_output")

def test():
    test_noinfluence()
    test_highinfluence()

def test_noinfluence():
    ExMockModelPredict1 = MockModelPredict1()
    trained_model = ClassifierModelVisitor(ExMockModelPredict1, ExMockModelPredict1.predict, 1)
    auditor = Auditor(modelvisitor=trained_model)
    df = pandas.DataFrame({"a": [1.0,2.0,3.0,4.0]})
    y_df = pandas.DataFrame({"b": [0,0,0,0]})
    data = load_testdf_only(df, y_df)
    auditor(data)
    ranks = auditor._audits_data["ranks"]
    print("pretrained model, no influence rank correct? --", ranks[0] == ('a',0.0))

def test_highinfluence():
    ExMockModelPredict1234 = MockModelPredict1234()
    trained_model = ClassifierModelVisitor(ExMockModelPredict1234, ExMockModelPredict1234.predict, 1)
    auditor = Auditor(modelvisitor=trained_model)
    df = pandas.DataFrame({"a": [1.0,2.0,3.0,4.0]})
    y_df = pandas.DataFrame({"b": [0,0,0,0]})
    data = load_testdf_only(df, y_df)
    auditor(data)
    ranks = auditor._audits_data["ranks"]
    print("pretrained model, high influence rank correct? --", ranks[0] == ('a',1.0))

class MockModelPredict1():
    def predict(self, X):
         return [1 for x in X]

class MockModelPredict1234():
    def predict(self, X):
         """
         Only predicts [0,0,0,0] if given [1,2,3,4].
         """
         if X[0] == [1.0] and X[1] == [2.0] and X[2] == [3.0] and X[3] == [4.0]:
              return [0,0,0,0]
         else:
              return [1,1,1,1]
         return prediction

if __name__ == "__main__":
#    german_example_audit()
    test()
