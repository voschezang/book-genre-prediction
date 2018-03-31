import os, sys, numpy as np
os.chdir('src/')  # fix for data.init_dataset()

import data, config, tfidf, models, sentimentanalysis
from utils import utils, io

# info = pandas.read_csv(config.dataset_dir + 'final_data.csv')
dataset = data.init_dataset()
dataset.info.keys()

# load model
m = config.dataset_dir + 'models/default_model.json'
w = config.dataset_dir + 'models/default_model_w.h5'
model = models.load_model(m, w)

if __name__ == '__main__':
    args = sys.argv
    if args:
        filename = '../' + args[1]
        # filename = config.dataset_dir + 'test/12.txt'

    tokens, lines = io.read_book3(filename)

    # build feature vector
    v1 = data.tokenlist_to_vector(tokens, dataset.sentiment_dataset)
    v2 = np.array(sentimentanalysis.per_book(lines))
    x = np.append(v1, v2)
    x_test = np.stack([x])

    # predict y
    y_test = model.predict(x_test)[0]
    results, best = data.decode_y(dataset, y_test, 6)

    print("-----------------------------\n\n")
    print("-- Results --")

    # print all values
    ls = list(results.items())
    ls.sort(key=lambda x: x[1], reverse=True)
    th = ['Genre', 'Score']
    rows = []
    for k, v in ls:
        rows.append([k, utils.format_score(v)])
    print(utils.gen_table(th, rows))

    print('\n Top 5 genres:')
    th = ['#', 'Genre']
    rows = []
    for i, v in enumerate(best):
        rows.append([str(i + 1), v])
    print(utils.gen_table(th, rows))

    print('\n - \n')
    v = utils.format_score(results[best[0]])
    print('Predicted genre: "%s" with a score of %s%s \n\n' % (best[0], v,
                                                               '%'))
