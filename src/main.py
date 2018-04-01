import os, sys, numpy as np
import config
os.chdir('src/')  # fix for data.init_dataset()
np.random.seed(config.seed)

import data, tfidf, models, sentimentanalysis
from utils import utils, io

# info = pandas.read_csv(config.dataset_dir + 'final_data.csv')
dataset = data.init_dataset()

# load model
m = config.dataset_dir + 'models/default_model.json'
w = config.dataset_dir + 'models/default_model_w.h5'
model = models.load_model(m, w)

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        filename = '../' + args[1]
    else:
        filename = config.dataset_dir + '1118.txt'

    print('\n filename:', filename)
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

    # for x in range(3):
    #     # import os, sys, numpy as np
    #     # import config
    #     # np.random.seed(config.seed)
    #     # print("NP - - -", np.random.random(2))

    #     # import data, tfidf, models, sentimentanalysis
    #     # from utils import utils, io
    #     # dataset = data.init_dataset()
    #     # tokens, lines = io.read_book3(filename)

    #     v1_ = data.tokenlist_to_vector(tokens, dataset.sentiment_dataset)
    #     print("V - :::::", v1_[:20])
    #     for i, val in enumerate(v1):
    #         if not val == v1_[i]:
    #             print('not eq', val, v1_[i])

    #     v1 = data.tokenlist_to_vector(tokens, dataset.sentiment_dataset)
    #     v2 = np.array(sentimentanalysis.per_book(lines))
    #     x = np.append(v1, v2)
    #     x_test = np.stack([x])
    #     model = models.load_model(m, w)
    #     y_test = model.predict(x_test)[0]
    #     results, best = data.decode_y(dataset, y_test, 6)
    #     print(best)
