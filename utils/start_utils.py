import inspect
import sklearn.datasets as ds
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

def get_classifiers():
    classifiers = all_estimators(type_filter='classifier')
    return sorted([name for name, cls in classifiers if issubclass(cls, ClassifierMixin)])

def get_classifier_map():
    classifiers = all_estimators(type_filter='classifier')
    return {name: cls for name, cls in classifiers if issubclass(cls, ClassifierMixin)}

def get_datasets():
    datasets = []
    for name, func in inspect.getmembers(ds, inspect.isfunction):
        if name.startswith("load_"):
            sig = inspect.signature(func)
            if "return_X_y" in sig.parameters:
                try:
                    func(return_X_y=True)
                    datasets.append(name)
                except:
                    pass
    return sorted(datasets)

def load_sample_dataset(loader_name):
    loader_func = getattr(ds, loader_name)
    data = loader_func(as_frame=True)
    return data.frame
