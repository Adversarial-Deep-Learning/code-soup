class AttackException(Exception):
    pass

class WordNotInDictionaryException(AttackException):
    pass

class UnknownPOSException(AttackException):
    pass