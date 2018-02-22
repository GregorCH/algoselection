class Error(Exception):
    '''Base class for exceptions in this module.'''
    pass

class SGMNegativeValueError(Error):
    '''TODO'''
    def __init__(self, expression = None,
        message = 'Found value less then 0 after performing shift operation.\
                   Consider using greater shift value.'):
        self.expression = expression
        self.message = message

class IncopatibleInputFile(Error):
    '''TODO'''
    def __init__(self, file, message):
        self.file = file
        self.message = message
