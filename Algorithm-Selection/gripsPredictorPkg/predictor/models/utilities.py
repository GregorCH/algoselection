import os
import _pickle as serializer

from .. import config, logger
from ..performance import measurement as pm

log = logger._Logger.get_logger(__name__) # set module name for logging
cfg = config.global_config

def save_classifier(classifier, filename):
    '''
    .. warning TODO Description.
    '''
    fullpath = os.path.join(cfg.models_dir, filename + '.model')
    with open(fullpath, 'wb') as output:
        serializer.dump(classifier, output)
        log.info('Classifier %s saved on path: %s.' % (filename + '.model', fullpath))

def load_classifier(filename):
    '''
    .. warning TODO Description.
    '''
    fullpath = os.path.join(cfg.models_dir, filename + '.model')
    with open(fullpath, 'rb') as input:
        classifier = serializer.load(input)
        log.info('Classifier "%s" loaded from path "%s".' % (filename + '.model', fullpath))
