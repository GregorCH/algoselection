import os, errno, logging

LOGS_FILE_PATH = os.path.abspath(
                     os.path.join(
                        os.path.dirname(__file__),
                        '../logs'
                     )
                 )

class _ConsoleDebugMsgFilter():
    def filter(self, record):
        return 0 if record.levelname == 'DEBUG' else 1

class _Logger(object):
    '''TODO
    '''

    predictor_logger = None

    def __init__(self):
        '''Initializes logger for logging messages in log file and console. This
        method should not never been explicitly called. Instead, ``get_logger`` method
        should be used when obtaining reference to logger instance.

        Messages of level INFO and higher (except those of DEBUG level) are logged
        on the console. In the log file are logged messages of level DEBUG and
        higher.
        '''

        # create logger
        self.predictor_logger = logging.getLogger('predictor')
        self.predictor_logger.setLevel(logging.DEBUG)

        # create logs dir if it doesn't exist
        if not os.path.exists(LOGS_FILE_PATH):
            try:
                os.makedirs(LOGS_FILE_PATH)
            except OSError as e:
                if e.errno != errno.EEXIST: # don't bother if directory is created between checking if it exists and mkdirs call
                    raise

        # create file handler for logs
        fh = logging.FileHandler(os.path.join(LOGS_FILE_PATH, 'predictor.log'))
        fh.setLevel(logging.DEBUG)
        fformatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        fh.setFormatter(fformatter)

        # create console handler for logs
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        cformatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        ch.setFormatter(cformatter)
        ch.addFilter(_ConsoleDebugMsgFilter())

        # add ch to logger
        self.predictor_logger.addHandler(fh)
        self.predictor_logger.addHandler(ch)

    @classmethod
    def get_logger( cls, name ):
        '''Returns object of logger class.

        If logger object does not exist, it is created and returned, otherwise
        existing logger instance is returned.
        '''
        if ( cls.predictor_logger is None ):
            cls.predictor_logger = _Logger()
        cls.predictor_logger = logging.getLogger(name)
        return cls.predictor_logger
