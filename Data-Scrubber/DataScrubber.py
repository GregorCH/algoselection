
import pandas as pd   
import numpy as np

DATA_PATH = 'rawdata/'				#where the raw data files are
ALT_OUTPUT_PATH = 'alt_output/'		#there will be many files produced -- the files that are not "the main ones"
									# are placed in this directory

feasDF = pd.read_csv(DATA_PATH+"mipdev_feasibility.csv")       #read the feasibilty file into a data frame
feasible_DF = feasDF[feasDF['Code'] != 'inf'].groupby('Problem Name').count()
infeasible_DF = feasDF[feasDF['Code'] == 'inf'].groupby('Problem Name').count()
feasible_instances = set(feasible_DF.index)
infeasible_instances = set(infeasible_DF.index)


#some of the instances do not have a complete set of labels
# (all are missing data from either emphasis_optimality or seperate_aggressive or both)
# and when a seed is missing any time data at all, the PDInt is also missing
instances_with_incomplete_data = set([	'buildingenergy',
										'neos-565672',
										'neos-848589',
										'neos-872648',
										'neos-873061',
										'netdiversion',
										'npmv07',
										'ns1758913',
										'ofi',
										'opm2-z11-s8',
										'sing245',
										'wnq-n100-mw99-14'])

feasible_instances = feasible_instances.difference(instances_with_incomplete_data)
infeasible_instances = infeasible_instances.difference(instances_with_incomplete_data)


STATUS_COLS = range(1,15)          #which columns contain the completion status of an algorithm
DATA_COLS = range(15,29)          #which columns contain the PD Ints
COL_OFFSET = 14                    #the number of columns offsetting the completion status from PD Ints


PD = pd.read_csv(DATA_PATH+"mipdev_integral_data.csv")        #read the PD Int file into a data frame
PD = PD.drop([0,1])           						#remove the extraneous header rows
PD = PD.drop(PD.columns[1], axis=1)   				#drop seed #


Times = pd.read_csv(DATA_PATH+"mipdev_time_data.csv")			#read the time data file into a data frame
Times = Times.drop([0,1])            				#remove the extraneous header rows
Times = Times.drop(Times.columns[1], axis=1)   		#drop seed #
#in the time data files, the status and data columns are opposite that of the PDInt file for some reason
# the next line switches them back so that the status cols are in front of the data cols
Times = Times[[Times.columns[0]]+Times.columns[DATA_COLS].tolist()+Times.columns[STATUS_COLS].tolist()]


Regions = pd.read_csv(DATA_PATH+"regions_time_data.csv")		#read the time data file into a data frame
Regions = Regions.drop([0,1])            			#remove the extraneous header rows
Regions = Regions.drop([Regions.columns[1],'emphasis_cpsolver','emphasis_cpsolver.1'], axis=1)
#in the time data files, the status and data columns are opposite that of the PDInt file for some reason
# the next line switches them back so that the status cols are in front of the data cols
Regions = Regions[[Regions.columns[0]]+Regions.columns[DATA_COLS].tolist()+Regions.columns[STATUS_COLS].tolist()]											#drop seed #, also drop the cpsolver data since it is all 600

RegionsPD = pd.read_csv(DATA_PATH+"regions_integral_data.csv")
RegionsPD = RegionsPD.drop([0,1])
RegionsPD = RegionsPD.drop([RegionsPD.columns[1],'emphasis_cpsolver','emphasis_cpsolver.1'], axis=1)


######the following data_frames contain data provided to us separately by Robert's colleague######
REDCOST_STATUS_COLS = range(1,3)          #which columns contain the completion status of an algorithm
REDCOST_DATA_COLS = range(3,5)          #which columns contain the PD Ints
REDCOST_COL_OFFSET = 2                     #the number of columns offsetting the completion status from PD Ints

RedCost = pd.read_csv(DATA_PATH+"RedCost_time_data.csv")			#read the time data file into a data frame
RedCost = RedCost.drop([0,1])            						#remove the extraneous header rows
RedCost = RedCost.drop(RedCost.columns[1], axis=1)   	#drop seed #
#in the time data files, the status and data columns are opposite that of the PDInt file for some reason
# the next line switches them back so that the status cols are in front of the data cols
RedCost = RedCost[[RedCost.columns[0]]+RedCost.columns[REDCOST_DATA_COLS].tolist()+RedCost.columns[REDCOST_STATUS_COLS].tolist()]	

RedCostPD = pd.read_csv(DATA_PATH+"RedCost_integral_data.csv")
RedCostPD = RedCostPD.drop([0,1])
RedCostPD = RedCostPD.drop(RedCostPD.columns[1], axis=1)   	#drop seed #


RedCost=RedCost.rename(columns = {RedCost.columns[0]:'Problem Name'})
RedCost.columns = RedCost.columns.str.replace('.1','_Status')

RedCostPD=RedCostPD.rename(columns = {RedCostPD.columns[0]:'Problem Name'})
RedCostPD.columns = [RedCostPD.columns[0]] + [x+'_Status' for x in RedCostPD.columns[REDCOST_STATUS_COLS]] + \
				 [x for x in RedCostPD.columns[REDCOST_DATA_COLS]]
RedCostPD.columns = RedCostPD.columns.str.replace('.1','')

##################################################################################################





#rename the data_frame columns to correct names
PD=PD.rename(columns = {PD.columns[0]:'Problem Name'})
PD.columns = [PD.columns[0]] + [x+'_Status' for x in PD.columns[STATUS_COLS]] + \
				 [x for x in PD.columns[DATA_COLS]]
PD.columns = PD.columns.str.replace('.1','')

RegionsPD=RegionsPD.rename(columns = {RegionsPD.columns[0]:'Problem Name'})
RegionsPD.columns = [RegionsPD.columns[0]] + [x+'_Status' for x in RegionsPD.columns[STATUS_COLS]] + \
				 [x for x in RegionsPD.columns[DATA_COLS]]
RegionsPD.columns = RegionsPD.columns.str.replace('.1','')

Times=Times.rename(columns = {Times.columns[0]:'Problem Name'})
Times.columns = Times.columns.str.replace('.1','_Status')

Regions=Regions.rename(columns = {Regions.columns[0]:'Problem Name'})
Regions.columns = Regions.columns.str.replace('.1','_Status')


# changing the type of the PD int columns and Times columns
for i in DATA_COLS: # changing for numerics 
	PD[[PD.columns[i]]] = PD[[PD.columns[i]]].apply(pd.to_numeric)
	RegionsPD[[RegionsPD.columns[i]]] = RegionsPD[[RegionsPD.columns[i]]].apply(pd.to_numeric)
	Times[[Times.columns[i]]] = Times[[Times.columns[i]]].apply(pd.to_numeric)
	Regions[[Regions.columns[i]]] = Regions[[Regions.columns[i]]].apply(pd.to_numeric)

for i in REDCOST_DATA_COLS: # changing for numerics 
	RedCostPD[[RedCostPD.columns[i]]] = RedCostPD[[RedCostPD.columns[i]]].apply(pd.to_numeric)
	RedCost[[RedCost.columns[i]]] = RedCost[[RedCost.columns[i]]].apply(pd.to_numeric)


#some of the entries are more specific than we need them to be -- this code puts everything into three categories:
# 'ok','fail', and 'timelimit'




for i in REDCOST_STATUS_COLS: #changing the names to ok, fail and timelimit
   	#change anything that is not 'ok','fail', or 'timelimit' into one of these
	#
	RedCostPD[RedCostPD.columns[i]] = RedCostPD[RedCostPD.columns[i]].replace(['memlimit'], 'fail')
	RedCost[RedCost.columns[i]] = RedCost[RedCost.columns[i]].replace(['memlimit'], 'fail')
	#
	#change the type of the column to categorical for faster access/less memory usage
	RedCostPD[RedCostPD.columns[i]] = RedCostPD[RedCostPD.columns[i]].astype('category')
	RedCost[RedCost.columns[i]] = RedCost[RedCost.columns[i]].astype('category')
	
	#algorithm should not average in otherwise fast fail times with slower success times
	RedCostPD.loc[RedCostPD[RedCostPD.columns[i]] == 'fail', RedCostPD.columns[i+REDCOST_COL_OFFSET]] = 360000.0
	RedCost.loc[RedCost[RedCost.columns[i]] == 'fail', RedCost.columns[i+REDCOST_COL_OFFSET]] = 3600.0

	


def AggregateOnString(df, trigger_string):# calculate the number of instances for one of the solution desciptions:
										  #  (e.g. ok, timelimit, or fail)
	#initialize with just the problem names (dataframe is otherwise empty)
	Num_Triggered = df[['Problem Name']].groupby('Problem Name').count()

	#load up the new data frame with the number of solved instances
	for i in STATUS_COLS:
		column = df.columns[i]
		
		#carve out just the problem names and their results on the specific algorithm
		sliced_df = df[[column,'Problem Name']]
		
		#group up by the problem name and count how many strings match the trigger
		temp_DF = sliced_df[sliced_df[column] == trigger_string].groupby('Problem Name').count()
		
		#add this information into the return data frame
		Num_Triggered[column] = temp_DF[column]
    
    #replace all NaN with 0 (NaN occurs if there is no positive count)
	Num_Triggered[np.isnan(Num_Triggered)] = 0

	return Num_Triggered

def avg_PD_if_string(df,trigger_string):  # calculate avg over seeds for one of the solution desciption- ok timelimit and fail
	result = pd.DataFrame() 
	for i in DATA_COLS:
		sliced_df = df[['Problem Name',df.columns[i-COL_OFFSET],df.columns[i]]]
		a=sliced_df.loc[sliced_df[sliced_df.columns[1]] == trigger_string]
		b= a.groupby(['Problem Name']).mean()
		result[b.columns[0]] = b[b.columns[0]]
	return result

def cap_values(df, cols, upper_bound):		#takes all values above a certain number and caps them at the upper bound
									# (e.g. takes all values above 60k and sets them to 60k)
	for i in cols:
		column = df.columns[i]
		df.loc[df[column] > upper_bound,column] = upper_bound



cap_values(PD,DATA_COLS,60000)
cap_values(Times,DATA_COLS,600)
cap_values(Regions,DATA_COLS,600)

cap_values(RedCostPD,REDCOST_DATA_COLS,360000)
cap_values(RedCost,REDCOST_DATA_COLS,3600)


#separate the time data into completion time for feasible and infeasible training instances
bool_feasible_Times = [entry in feasible_instances for entry in Times['Problem Name']]
Times_feasible = Times[bool_feasible_Times]
Times_infeasible = Times[[not entry for entry in bool_feasible_Times]]




#only aggregate the PD Integrals on the instances that were feasible (since infeasible instances always report 60.000 PDInt)
bool_feasible_PD = [entry in feasible_instances for entry in PD['Problem Name']]
PD_feasible = PD[bool_feasible_PD]



#our initial examination of the redcost data set will be limited to the feasible instances since we
# do not have enough infeasible instances to interpolate with/extrapolate from
bool_feasible_RedCostPD = [entry in feasible_instances for entry in RedCostPD['Problem Name']]
RedCostPD = RedCostPD[bool_feasible_RedCostPD]
bool_feasible_RedCost = [entry in feasible_instances for entry in RedCost['Problem Name']]
RedCost = RedCost[bool_feasible_RedCost]






#output the Time data
avg_feasible_time = Times_feasible.groupby(['Problem Name']).mean()
avg_infeasible_time = Times_infeasible.groupby(['Problem Name']).mean()

avg_feasible_time.to_csv('mipdev_feasible_avg_time.csv', sep=',')
avg_infeasible_time.to_csv('mipdev_infeasible_avg_time.csv', sep=',')







#output the PD Integrals regardless of algorithm outcome
avg_PD = PD_feasible.groupby(['Problem Name']).mean()
avg_PD.to_csv('mipdev_avg_PD.csv',sep=',')

regions_avg_time = Regions.groupby(['Problem Name']).mean()
regions_avg_time.to_csv('regions_avg_time.csv',sep=',')

regions_avg_PD = RegionsPD.groupby(['Problem Name']).mean()
regions_avg_PD.to_csv('regions_avg_PD.csv',sep=',')

RedCost_avg_time = RedCost.groupby(['Problem Name']).mean()
RedCost_avg_time.to_csv('RedCost_avg_time.csv',sep=',')

RedCost_avg_PD = RedCostPD.groupby(['Problem Name']).mean()
RedCost_avg_PD.to_csv('RedCost_avg_PD.csv',sep=',')




#print the counts (out of 5) of the various possible outcomes
count_feasible_timelimit = AggregateOnString(Times_feasible, 'timelimit')
count_feasible_ok = AggregateOnString(Times_feasible, 'ok')
count_feasible_fail = AggregateOnString(Times_feasible, 'fail')

count_infeasible_timelimit = AggregateOnString(Times_infeasible, 'timelimit')
count_infeasible_ok = AggregateOnString(Times_infeasible, 'ok')
count_infeasible_fail = AggregateOnString(Times_infeasible, 'fail')

regions_ok = AggregateOnString(Regions,'ok')
regions_timelimit = AggregateOnString(Regions,'timelimit')

count_feasible_timelimit.to_csv(ALT_OUTPUT_PATH+'mipdev_feasible_count_timelimit.csv', sep=',')
count_feasible_ok.to_csv(ALT_OUTPUT_PATH+'mipdev_feasible_count_ok.csv', sep=',')
count_feasible_fail.to_csv(ALT_OUTPUT_PATH+'mipdev_feasible_count_fail.csv', sep=',')

count_infeasible_timelimit.to_csv(ALT_OUTPUT_PATH+'mipdev_infeasible_count_timelimit.csv', sep=',')
count_infeasible_ok.to_csv(ALT_OUTPUT_PATH+'mipdev_infeasible_count_ok.csv', sep=',')
count_infeasible_fail.to_csv(ALT_OUTPUT_PATH+'mipdev_infeasible_count_fail.csv', sep=',')

regions_ok.to_csv(ALT_OUTPUT_PATH+'regions_count_ok.csv',sep=',')
regions_timelimit.to_csv(ALT_OUTPUT_PATH+'regions_count_timelimit.csv',sep=',')



#output the PD Integrals only if timeout
avg_PDInt = avg_PD_if_string(PD_feasible, 'timelimit')
avg_PDInt.to_csv(ALT_OUTPUT_PATH+'mipdev_avg_PD_if_timelimit.csv', sep=',')