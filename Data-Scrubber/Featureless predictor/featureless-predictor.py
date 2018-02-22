import sys
import csv
import pandas as pd
import numpy as np

#this is our shifted geometric mean code (contained in a separate file)
import sgm

#make sure the correct command line format is used
if (len(sys.argv) != 3):
	print('Incorrect call format. Try: featureless-predictor.py <data_file> <\'time\' or \'pdi\'>')
	exit(1)

#full path to the file which contains time or PDI data
LABELS_FILE    = sys.argv[1]

#second parameter is either 'time' or 'PDInt' -- used for the shift in the shifted geometric mean
if sys.argv[2] == 'time':
	ALPHA = 10
elif sys.argv[2] == 'pdi':
	ALPHA = 1000
else:
	print('Second argument must be one of \'time\' or \'pdi\' (case sensitive).  Received: \'' + sys.argv[2] + '\'')
	exit(2)

#read the labels file, create a data frame
labels = pd.read_csv(LABELS_FILE, index_col = 0)

#at first, there is no portfolio.  We will append to this as we go.
portfolio = []

#at first, all algorithms are available to choose for the portfolio
# over time we will remove them one by one as they are chosen
algorithms_left = {column for column in labels}

#for debugging purposes can uncomment these lines
#print('Based on:', LABELS_FILE)
#average = {algorithm : sgm.shifted_geometric_mean(labels[algorithm], ALPHA) for algorithm in algorithms_left}
#print(average)

#loop until we have put all the algorithms into our portfolio (i.e. terminate when algorithms_left is empty)
while(algorithms_left):
	#calculate shifted geometric mean for each algorithm (as a dictionary)
	average = {algorithm : sgm.shifted_geometric_mean(labels[algorithm], ALPHA) for algorithm in algorithms_left}

	#get the algorithm with the lowest sgm (greedy choice)
	chosen_algorithm = min(average, key=average.get)

	#append this algorithm to our portfolio
	portfolio.append(chosen_algorithm)

	#remove the algorithm from the algorithm list so that we do not choose it again
	algorithms_left.remove(chosen_algorithm)

	#now that we have chosen this algorithm to be in our portfolio, it is not possible in future choices to do worse
	# than our existing portfolio.  To model this change, we take the current algorithm's perfomance to be the new
	# upper bound.  To include this in our computations, we modify the existing columns with the new upper bound
	for algorithm in algorithms_left:
		labels[algorithm] = labels[[chosen_algorithm, algorithm]].min(axis=1)
		
#in case you want to verbosely output the portfolio to the console directly, you can uncomment this next line
#
#print('Based on:', LABELS_FILE, ':\n', portfolio,'\n')

#but to match the output of the rest of our team, we output the results as a CSV file listing
# the portfolio for every instance separately
output_filename = 'results/featureless_portfolio_' + LABELS_FILE

output_file = open(output_filename, "w", newline='')    #CAUTION: only works in Python 3!!!!
writer = csv.writer(output_file)
for index,_ in labels.iterrows():
	writer.writerow([index] + portfolio)