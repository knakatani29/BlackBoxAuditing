from abc import ABCMeta, abstractmethod
import time

class AbstractModelVisitor(object, metaclass=ABCMeta):
  def __init__(self, model_name):

    """
    ModelVisitor class enables you to use a pre-trained model for auditing. 
    Users can either:
    1) create a model factory and build it.
    2) use ClassifierModelVisitor class in ClassifierModelVisitor.py.

    ModelVisitor needs to have a test method that returns prediction for each test data.
    """

    self.model_name = "{}".format(time.time())

  @abstractmethod
  def test(self, test_set):
    pass
