import sys, os
import numpy as np
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format
cmap = sns.light_palette("green", as_cmap=True)

class EvaluationSummarizer(object):

    def __init__(self, input_path, output_path, detail_postfix, summary_postfix):

        self.input_path = input_path
        self.output_path = output_path
        self.detail_postfix = detail_postfix
        self.summary_postfix = summary_postfix
        self.detail_list = self.genFileNameList(detail_postfix)
        self.summary_list = self.genFileNameList(summary_postfix)

        self.genSummaryTable()
        self.genRelativeSummaryFigure()

    def genFileNameList(self,postfix):
        list = []
        for file in os.listdir(self.input_path):
            if file.endswith(postfix):
                print(file)
                list.append(file)
        return list

    def genSummaryTable(self):
        table_contents = []
        for f in self.summary_list:
            df = pd.read_csv(f, sep='\t')
            table_contents.append(df)
        table = pd.concat(table_contents)
        colnames = list(table)
        table.sort_values(colnames[0:1])
        with open(self.output_path+"/resultsOverview.tex", 'w') as tf:
            tf.write(table.to_latex(index=False))


    def genRelativeSummaryFigure(self):
        figure_contents = []
        for f in self.detail_list:
            pred = np.load(f)
            res = (pred[:,0]==pred[:,1])
            config_name = f.split("out_")[1].split("_predictions")[0]
            df = pd.DataFrame({"question_index":range(0,len(res)),
                               "res":res,
                               "configuration":config_name})
            figure_contents.append(df)
        long_table = pd.concat(figure_contents)
        wide_table= long_table.pivot(index="question_index",columns="configuration",values="res")
        plt.figure(figsize=(10, 10))
        sns.heatmap(wide_table, cmap=cmap, cbar=False, yticklabels=False)
        plt.xticks(rotation=45)
        plt.savefig(self.output_path+'/figsummary.png')


def main(argv):
    parser = argparse.ArgumentParser(description='Homework 3 QA pipeline with learning - evaluation summary')
    parser.add_argument('--input-path', type=str,
                        default=".",
                        help='Path to evaluation summary input folder')
    parser.add_argument('--out-path', type=str,
                        default=".",
                        help='Path to evalation summary output folder')
    parser.add_argument('--detail-postfix', type=str,
                        default="_detailed_predictions.npy",
                        help='Postfix of files containing detailed input')
    parser.add_argument('--summary-postfix', type=str,
                        default="_summary_predictions.txt",
                        help='Postfix of files containing summary input')

    args = parser.parse_args(argv)
    print(args)

    EvaluationSummarizer(args.input_path, args.out_path,
                         args.detail_postfix, args.summary_postfix)

if __name__ == '__main__':
    main(sys.argv[1:])


