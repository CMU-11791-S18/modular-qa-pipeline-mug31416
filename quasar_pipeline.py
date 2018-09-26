import sys
import json
import re
import numpy as np
import argparse
from sklearn.externals import joblib
import math
from collections import Counter

from Retrieval import Retrieval
from CountFeaturizer import CountFeaturizer
from TfIdfFeaturizer import TfIdfFeaturizer
from MultinomialNaiveBayes import MultinomialNaiveBayes
from MLP import MLP
from SVM import SVM
from Evaluator import Evaluator


STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', 'inc', 'll', 'i'))

class Pipeline(object):

    def __init__(self, trainFilePath, valFilePath, valResultsPath,
                 snippet_type, top,
                 retrievalInstance, featurizerInstance, classifierInstance, debug):

        self.debug = debug
        self.snippet_type = snippet_type
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance

        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()

        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()

        self.valResultsPath = valResultsPath
        self.top = top
        self.question_answering()


    def stemAndStopText(self, text):
        res = []

        for w in text.lower().split():
            if not w in STOP_WORDS:
                res.append(self.stemmer.stem(w))

        return ' '.join(res)

    @staticmethod
    def cleanText(text):
        """
        Method to map a fixed list of higher frequency category names
        to names that will not be split by a tokenizer.

        :param text: input string
        :return: cleaned string
        """

        transform_list = [(r"'s\b",' '),
                          (r"(^|\W)[.]net(\W|$)","\g<1>dotnet\\2"),
                          (r"(^|\W)c[+][+](\W|$)", "\g<1>cplusplus\\2"),
                          (r"(^|\W)objective[-]c(\W|$)", "\g<1>objectivec\\2"),
                          (r"(^|\W)node[.]js(\W|$)", "\g<1>nodedotjs\\2"),
                          (r"(^|\W)asp[.]net(\W|$)", "\g<1>aspdotnet\\2"),
                          (r"(^|\W)e[-]commerce(\W|$)", "\g<1>ecommerce\\2"),
                          (r"(^|\W)java[-]ee(\W|$)", "\g<1>javaee\\2"),
                          (r"(^|\W)32[-]bit(\W|$)", "\g<1>32bit\\2"),
                          (r"(^|\W)kendo[-]ui(\W|$)", "\g<1>kendoui\\2"),
                          (r"(^|\W)jquery[-]ui(\W|$)", "\g<1>jqueryui\\2"),
                          (r"(^|\W)c[+][+]11(\W|$)", "\g<1>cplusplus11\\2"),
                          (r"(^|\W)windows[-]8(\W|$)", "\g<1>windows8\\2"),
                          (r"(^|\W)ip[-]address(\W|$)", "\g<1>ipaddress\\2"),
                          (r"(^|\W)backbone[.]js(\W|$)", "\g<1>backbonedotjs\\2"),
                          (r"(^|\W)angular[.]js(\W|$)", "\g<1>angulardotjs\\2"),
                          (r"(^|\W)as[-]if(\W|$)", "\g<1>asif\\2"),
                          (r"(^|\W)actionscript[-]3(\W|$)", "\g<1>actionscript3\\2"),
                          (r"(^|\W)[@]placeholder(\W|$)", "\g<1>atsymbolplaceholder\\2"),
                          (r"\W+", " ")
                          ]
        return_text = text.lower()
        for p in transform_list:
            return_text = re.sub(p[0], p[1], return_text)

        return return_text

    @staticmethod
    def computeOccurQty(needle, haystack):
        tmp = re.findall(r'\b'+needle+r'\b', haystack)

        return len(tmp)

    @staticmethod
    def transformData(snippets_clean, freq_answ):
        '''
        The method concatenates snippets that contain at least
        one frequent candidate answer.

        :param snippets_clean: "clean" snippets
        :param freq_answ: a set/list of frequent candidate answers.
        :return:
        '''

        select_snippets = set()

        for answer in freq_answ:

            clean_answer = Pipeline.cleanText(answer)

            for p in snippets_clean:
                occurrQty = Pipeline.computeOccurQty(clean_answer, p)
                if occurrQty > 0:
                    select_snippets.add(p)

        return ' '.join(select_snippets)


    # method is modified from the code pase to add cleaning and
    # transformation steps
    def makeXY(self, dataQuestions, freq_answ, is_train):
        X = []
        Y = []

        quest_id = 0
        for question in dataQuestions:
            snippets = None
            if self.snippet_type == 'short':
                if self.debug:
                    print('Using short snippets')
                snippets = self.retrievalInstance.getShortSnippetsList(question)
            elif self.snippet_type == 'long':
                if self.debug:
                    print('Using long snippets')
                snippets = self.retrievalInstance.getLongSnippetsList(question)

            query = self.retrievalInstance.getQuery(question)
            answer = self.retrievalInstance.getAnswer(question)

            if self.debug:
                print("==============================")
                print(quest_id, '===>', query)
                print(quest_id, '===>', answer)

            skip_flag = False
            if is_train:
                if not answer in freq_answ:
                    if self.debug:
                        print('Skipping non-frequent answer: %s' % answer)
                    skip_flag=True

            if not skip_flag:
                # clean all text
                query_clean = Pipeline.cleanText(query)
                if snippets is None:
                    if self.debug:
                        print('Using no snippets')
                    select_snippets = ''
                else:
                    snippets_clean = [Pipeline.cleanText(t) for t in snippets]
                    #select_snippets = ' '.join(snippets_clean)
                    select_snippets = Pipeline.transformData(snippets_clean, freq_answ)

                X.append(query_clean + ' '+ select_snippets)
                Y.append(answer)

            quest_id = quest_id + 1

        return X, Y

    # method modified from code base:
    # 1) identifies top-X most frequent answer categories
    # 2) outputs results tables
    def question_answering(self):

        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        cand_qty = len(candidate_answers)

        # Identify most frequent answer categories
        answFreq = dict()
        for i in range(cand_qty):
            answFreq[candidate_answers[i]] = 0

        quest_qty = float(len(self.trainData['questions']))
        for question in self.trainData['questions']:
            answ = self.retrievalInstance.getAnswer(question)
            answFreq[answ] = answFreq[answ] + 1

        answProb = dict()
        for answ in answFreq:
            answProb[answ] = answFreq[answ]/quest_qty

        tmp = [(v, k) for k, v in answProb.items()]
        tmp.sort(reverse=True)
        s = 0
        freq_answ = set()
        for i in range(self.top):
            prob, answ = tmp[i]
            s = s + prob
            freq_answ.add(answ)
            if self.debug:
                print('%s %g' % (answ, prob))

        if self.debug:
            print('Top %d sum %g freq answ set qty %d' % (self.top, s, len(freq_answ)))

        X_train, Y_train = self.makeXY(self.trainData['questions'][0:10000], freq_answ, is_train=True)
        X_val, Y_val_true = self.makeXY(self.valData['questions'][0:500], freq_answ, is_train=False)

        # featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(X_train, X_val)
        self.clf = self.classifierInstance.buildClassifier(X_features_train, Y_train)

        # Prediction
        # Added processing steps to convert from binary to multi-class prediction representation
        Y_val_pred = self.clf.predict(X_features_val)
        pred_class = np.array(Y_val_pred)
        true_class = np.array(Y_val_true)

        if self.debug:
            print('X_features_val.shape:', X_features_val.shape)

            print(pred_class)
            print(true_class)
            print(np.mean(true_class == pred_class))

        # Evaluate
        self.evaluatorInstance = Evaluator()
        a =  self.evaluatorInstance.getAccuracy(true_class, pred_class)
        p,r,f = self.evaluatorInstance.getPRF(true_class, pred_class)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p)) #corrected from print("Precision: " + str(a))
        print("Recall: " + str(r)) #corrected from print("Recall: " + str(a))
        print("F-measure: " + str(f)) #corrected from print("F-measure: " + str(a))


        # Output files for further analysis
        with open('_'.join([self.valResultsPath,'summary_predictions.txt']), 'w') as file:
            file.write("Features" + "\t" + "Classifier" + "\t"
                    + "Accuracy" + "\t" + "Precision" + "\t"
                    + "Recall" + "\t" + "F-measure" + "\n")
            file.write(self.featurizerInstance.getName() + "\t" + self.classifierInstance.getName() + "\t"
                    + str(a) + "\t" + str(p) + "\t"
                    + str(r) + "\t" + str(f) + "\n")

        detailedRes = np.stack([true_class, pred_class], axis=1)
        np.save('_'.join([self.valResultsPath,'detailed_predictions.npy']), detailedRes)


def main(argv):
    parser = argparse.ArgumentParser(description='Homework 3 QA pipeline with learning - pipeline')
    parser.add_argument("--top-cat-limit",
                        type=int,
                        default=50,
                        help="Limit on the number of top categories to retain")
    parser.add_argument("--featurizer",
                        type=str,
                        choices =['tfidf','count'],
                        default='tfidf',
                        help="Featurizer option")
    parser.add_argument("--snippet_type",
                        type=str,
                        choices=['short', 'long', 'none'],
                        default='long',
                        help="snippet type")
    parser.add_argument("--model",
                        type=str,
                        choices =['mnb','svm','mlp'],
                        default='mnb',
                        help="Model option")
    parser.add_argument("--train-path",
                        type=str,
                        default="../datasets/quasar-s/quasar-s_train_formatted.json",
                        help="Path to training file")
    parser.add_argument('--valid-path', type=str,
                        default="../datasets/quasar-s/quasar-s_dev_formatted.json",
                        help='Path to training file')
    parser.add_argument('--out-path', type=str,
                        default=".",
                        help='Path to results file')
    parser.add_argument('--debug', type=bool,
                        default=False,
                        help='Debug print')

    args = parser.parse_args(argv)
    print(args)

    if args.featurizer=="tfidf":
        featurizerInstance = TfIdfFeaturizer(STOP_WORDS)
    elif args.featurizer=="count":
        featurizerInstance = CountFeaturizer(STOP_WORDS)
    else:
        raise Exception("Unknown featurizer")

    if args.model=="mnb":
        classifierInstance = MultinomialNaiveBayes()
    elif args.model=="svm":
        classifierInstance = SVM()
    elif args.model == "mlp":
        classifierInstance = MLP()
    else:
        raise Exception("Unknown model")

    outputFileStub = args.out_path+"/out_"+args.featurizer+"_"+args.model

    retrievalInstance = Retrieval()
    Pipeline(args.train_path, args.valid_path, outputFileStub,
             args.snippet_type,
             args.top_cat_limit,
             retrievalInstance,
             featurizerInstance, classifierInstance,
             args.debug)


if __name__ == '__main__':
    main(sys.argv[1:])

