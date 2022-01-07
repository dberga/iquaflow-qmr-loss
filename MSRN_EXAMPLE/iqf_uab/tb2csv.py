from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import glob
import os


def formatvals(vals, round_decimals = 5):
	newvals = []
	for val in vals:
		newval=str(round(val,round_decimals))
		newvals.append(newval)
	return newvals

def event2arrays(experiment_path):
	event_acc = EventAccumulator(experiment_path)
	event_acc.Reload()
	
	# scalars and histograms
	hists=event_acc.Tags()['histograms']
	scalars=event_acc.Tags()['scalars']
	
	# read and stack scalar values
	all_vals=[]
	for s in scalars:  # wall clock, number of steps and value for a scalar
		clock, steps, vals = zip(*event_acc.Scalars(s))
		vals = formatvals(vals)
		vals.insert(0,s) # scalar name at first row
		all_vals.append(vals)
	# to do: histograms

	return all_vals #all stacked values
		
def events2csv(dirpath = "./msrn/experiment"):
	experiment_paths=glob.glob(os.path.join(dirpath,'*'))
	for experiment_path in experiment_paths:
		if len(glob.glob(os.path.join(experiment_path,'events.out.tfevents*'))):
			experiment_name=os.path.dirname(experiment_path)
			all_vals = event2arrays(experiment_path)
			np.savetxt(
		        os.path.join(experiment_path, "events.csv"),
		        np.asarray(all_vals),
		        delimiter=",",
		        fmt='%s'
		    )
		else:
			print(experiment_path + "has no tfevents")
#def events2metriccsv(dirpath = "./msrn/experiment"):
# to do: stack all experiments for each metric separately (one csv per metric, all experiments in row)
