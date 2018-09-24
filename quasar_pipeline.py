import sys
import json
import re
import numpy as np
import argparse
from sklearn.externals import joblib
import math
from collections import Counter

from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from TfIdfFeaturizer import TfIdfFeaturizer
from Classifier import Classifier
from MultinomialNaiveBayes import MultinomialNaiveBayes
from SVM import SVM
from Evaluator import Evaluator


STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your', 'inc', 'll', 'i'))

class Pipeline(object):

    TOP_K_SNIPPETS = 5
    TRAIN_NEG_SAMPLE_QTY = 5

    def __init__(self, trainFilePath, valFilePath, valResultsPath, top,
                 retrievalInstance, featurizerInstance, classifierInstance):

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

    @staticmethod
    def cleanText(text):

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
    def makeHypos(query, candidates, freq_answ):

        hypos = []

        answ_id = 0
        for c in candidates:
            if answ_id in freq_answ:
                hypos.append((c, re.sub(r"(^|\W)atsymbolplaceholder(\W|$)", "\g<1>%s\\2" % c, query)))
            else:
                hypos.append((c, None))
            answ_id += 1

        return hypos

    @staticmethod
    def computeJaccardAndOverlap(answer,passage):
        passageset = set(passage.split())
        answerset = set(answer.split())
        intersect = answerset.intersection(passageset)

        return float(len(intersect))/max(len(passageset), len(answerset)), intersect

    @staticmethod
    def computeWeightJaccardAndOverlap(answer, passage, stop_words):
        passageDict = Counter(passage.split())
        answerset = set(answer.split())

        intersect = []
        qtyOverlap = 0.0
        passLen = 0.0
        for w, qty in passageDict.items():
            if w in stop_words:
                continue
            passLen += qty
            if w in answerset:
                qtyOverlap = qty
                intersect +=  qty * [w]

        return qtyOverlap / passLen, intersect

    @staticmethod
    def transformData(passages, hypos, true_answer, is_train):

        tmpRes = []
        INCLUDE_ALL = False

        answ_id = 0
        for answer, h in hypos:
            rel = 0
            transform_list = []

            if h is not None:
                rel = int(answer == true_answer)
                h = re.sub(r'\b%s\b' % answer, 'ATSYMBOLPLACEHOLDER', h)
                answer_set = set(answer.split())

                for p in passages:
                    if INCLUDE_ALL or p.find(answer) >= 0:
                        p = re.sub(r'\b%s\b' % answer, 'ATSYMBOLPLACEHOLDER', p)
                        score, intersect = Pipeline.computeWeightJaccardAndOverlap(h, p, STOP_WORDS)

                        transform_list.append((score, ' '.join(intersect)))

            answ_id += 1
            transform_list.sort(reverse=True)

            e = []
            for t in transform_list[0:Pipeline.TOP_K_SNIPPETS]:
                e.append(t[1])

            maxScore = transform_list[0][0] if transform_list else 0

            tmpRes.append( (rel, maxScore, ' '.join(e)))

        if is_train:
            tmpRes.sort(reverse=True)

        #print(is_train, tmpRes[0:5])
        relevance, evidence = [], []

        for r, _, e in tmpRes:
            relevance.append(r)
            evidence.append(e)

        return relevance, evidence


    def makeXY(self, dataQuestions, candidate_answers, freq_answ, is_train):
        X = []
        Y = []

        candidate_answers_clean = []

        for c in candidate_answers:
            candidate_answers_clean.append(Pipeline.cleanText(c))

        quest_id = 0
        for question in dataQuestions:

            #short_snippets = self.retrievalInstance.getShortSnippetsList(question)
            long_snippets = self.retrievalInstance.getLongSnippetsList(question)
            snippets = long_snippets
            query = self.retrievalInstance.getQuery(question)
            answer = self.retrievalInstance.getAnswer(question)

            # clean all text
            snippets_clean = [Pipeline.cleanText(t) for t in snippets]
            query_clean = Pipeline.cleanText(query)
            answer_clean = Pipeline.cleanText(answer)

            print("==============================")
            print(quest_id, '===>', query_clean)
            print(quest_id, '===>', answer_clean)

            relevance, evidence = Pipeline.transformData(snippets_clean,
                                                            Pipeline.makeHypos(query_clean,
                                                                               candidate_answers_clean,
                                                                               freq_answ),
                                                            answer_clean,
                                                            is_train)

            maxQty = len(evidence) if not is_train else 1 + Pipeline.TRAIN_NEG_SAMPLE_QTY
            X.extend(evidence[0:maxQty])
            Y.extend(relevance[0:maxQty])
            quest_id += 1

        return X, Y


    def question_answering(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        cand_qty = len(candidate_answers)

        answFreq = dict()
        for answId in range(cand_qty):
            answFreq[answId] = 0

        quest_qty = float(len(self.trainData['questions']))
        for question in self.trainData['questions']:
            answ = self.retrievalInstance.getAnswer(question)
            answId = None
            for i in range(cand_qty):
                if candidate_answers[i] == answ:
                    answId = i
                    break

            assert(answId is not None)
            answFreq[answId] += 1

        answProb = dict()

        for answId in range(cand_qty):
            answProb[answId] = answFreq[answId]/quest_qty

        boost = np.zeros(cand_qty)
        tmp = [(v, k) for k, v in answProb.items()]
        tmp.sort(reverse=True)
        s = 0
        freq_answ = set()
        for i in range(self.top):
            s += tmp[i][0]
            answ_id = tmp[i][1]
            freq_answ.add(answ_id)
            #boost[] = 1e5
        print('Top %d sum %g freq answ set qty %d' % (self.top, s, len(freq_answ)))


        X_train, Y_train = self.makeXY(self.trainData['questions'][0:100], candidate_answers,
                                       freq_answ, is_train=True)
        X_val, Y_val_true = self.makeXY(self.valData['questions'][0:100], candidate_answers,
                                       freq_answ, is_train=False)

        # featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(X_train, X_val)
        self.clf = self.classifierInstance.buildClassifier(X_features_train, Y_train)

        #Prediction
        Y_val_pred = np.array(self.clf.predict_log_proba(X_features_val))[:,1].reshape(-1, cand_qty)
        Y_val_true = np.array(Y_val_true).reshape(-1, cand_qty)

        Y_val_pred += boost
        pred_class = np.argmax(Y_val_pred, axis=1)
        true_class = np.argmax(Y_val_true, axis=1)

        print('Y_val_pred.shape:', Y_val_pred.shape)
        print('X_features_val.shape:', X_features_val.shape)
        print('Y_val_true.shape:', Y_val_true.shape)

        print(pred_class)
        print(true_class)
        print(np.mean(true_class == pred_class))

        self.evaluatorInstance = Evaluator()
        a =  self.evaluatorInstance.getAccuracy(true_class, pred_class)
        p,r,f = self.evaluatorInstance.getPRF(true_class, pred_class)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p)) #corrected from print("Precision: " + str(a))
        print("Recall: " + str(r)) #corrected from print("Recall: " + str(a))
        print("F-measure: " + str(f)) #corrected from print("F-measure: " + str(a))

        candidate_answers = np.array(candidate_answers)
        detailedRes = np.stack((candidate_answers[true_class],candidate_answers[pred_class]),axis=1)
        np.save('_'.join([self.valResultsPath,'detailed_predictions.npy']), detailedRes)


def main(argv):
    parser = argparse.ArgumentParser(description='Homework 3 QA pipeline with learning')
    parser.add_argument("--top-cat-limit",
                        type=int,
                        default=50,
                        help="Limit on the number of top categories to retain")
    parser.add_argument("--featurizer",
                        type=str,
                        choices =['tfidf','count'],
                        default='tfidf',
                        help="Featurizer option")
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

    args = parser.parse_args(argv)
    print(args)

    retrievalInstance = Retrieval()

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
    #elif args.model == "mlp":
    #    classifierInstance = MLP()
    else:
        raise Exception("Unknown model")

    outputFileStub = args.out_path+"/out_"+args.featurizer+"_"+args.model

    trainInstance = Pipeline(args.train_path, args.valid_path, outputFileStub,
                             args.top_cat_limit,
                             retrievalInstance, featurizerInstance, classifierInstance)


if __name__ == '__main__':
    main(sys.argv[1:])

