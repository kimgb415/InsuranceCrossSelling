from .opt import objective, space
from hyperopt import fmin, tpe
import json
from data.process import *

if __name__ == '__main__':
    train_df, dev_df, test_df = retrieve_train_dev_test_dataframe()

    best = fmin(
        fn=lambda params: objective(params, train_df, dev_df), 
        space=space, 
        algo=tpe.suggest, 
        max_evals=10
    )
    print(best) 
    # map dictionary best with the space dictionary to get the actual values

    with open('best_params.json', 'w') as f:
        json.dump(best, f)