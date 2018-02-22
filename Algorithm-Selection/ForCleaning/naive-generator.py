import sys
import pandas as pd
import numpy as np

def shifted_geometric_mean(iterable, alpha):
    """
    Calculates shifted geometric mean.

    :param iterable:        ordered collection of values which will be used for SGM calculation
    :param alpha:           value added to each element included in calculation of geometric mean

    :return                 value of shifted geometric mean
    """

    a = np.add(iterable, alpha);
    a = np.log(a)
    return np.exp(a.sum() / len(a)) - alpha


if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print('Incorrect call format. Try: naive-generator.py <actual_data_path> <predicted_data_path> <destionation_data_path> <time or pdi>')
        exit(1)

    ACTUAL_DATA_PATH    = sys.argv[1]
    PREDICTED_DATA_PATH = sys.argv[2]
    DESTINATION_PATH    = sys.argv[3]
    ALPHA               = 10 if sys.argv[4] == 'time' else 1000

    actual_data     = pd.read_csv(ACTUAL_DATA_PATH, index_col = None)
    predicted_data  = pd.read_csv(PREDICTED_DATA_PATH, header = None)

    # calculate shifted geometric mean for each algorithm in actual data
    average = {}
    actual_data_df = pd.DataFrame( actual_data )
    #print(actual_data_df)
    for column in actual_data_df.columns[1:]:
        print( actual_data_df[column] )
        average[column] = [ shifted_geometric_mean( iterable = actual_data_df[column], alpha = ALPHA ) ]

    # sort sgm and extract algorithm names according to new sorted values
    naive_algoritms = pd.DataFrame(average) \
                        .sort_values(by = 0, ascending = True, axis = 1) \
                        .columns

    # for each instance in test set set new portfolio
    naive_data = []
    for index, row in actual_data_df.iterrows():
        instance_name = row[0]
        curr_row = []
        curr_row.append(instance_name)
        curr_row.extend(list(naive_algoritms))
        naive_data.append(curr_row)

    output_df = pd.DataFrame( naive_data )
    output_df.to_csv('feature-independent-approach.csv', header = None, index = None)
