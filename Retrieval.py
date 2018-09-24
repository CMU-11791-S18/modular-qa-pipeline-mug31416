import abc
from abc import abstractmethod

class Retrieval:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self): #constructor for the abstract class
        pass

    @classmethod
    def getLongSnippets(self, question):
        longSnippets = question['contexts']['long_snippets']
        fullLongSnippets = ' '.join(longSnippets)
        return fullLongSnippets

    @classmethod
    def getLongSnippetsList(self, question):
        longSnippets = question['contexts']['long_snippets']
        return longSnippets

    @classmethod
    def getShortSnippets(self, question):
        shortSnippets = question['contexts']['short_snippets']
        fullShortSnippets = ' '.join(shortSnippets)
        return fullShortSnippets

    # adding extraction of individual short snippets
    @classmethod
    def getShortSnippetsList(self, question):
        shortSnippets = question['contexts']['short_snippets']
        return shortSnippets


    # adding extraction of query text
    @classmethod
    def getQuery(self, question):
        query = question['query']
        return query

    # adding extraction of query answer
    @classmethod
    def getAnswer(self, question):
        answer = question['answers'][0]
        return answer