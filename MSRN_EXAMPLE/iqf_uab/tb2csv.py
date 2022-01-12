from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from iq_tool_box.quality_metrics.benchmark import plot_1d
import numpy as np
import glob
import os


def formatvals(vals, round_decimals = 5, tostr=True):
	newvals = []
	for val in vals:
		if tostr:
			newval=str(round(val,round_decimals))
		else:
			newval=round(val,round_decimals)
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
	all_vals_num=[]
	for s in scalars:  # wall clock, number of steps and value for a scalar
		clock, steps, vals = zip(*event_acc.Scalars(s))
		vals_num = vals
		vals = formatvals(vals,True)
		vals.insert(0,s) # scalar name at first row
		all_vals.append(vals)
		all_vals_num.append(vals_num)
	# to do: histograms

	return all_vals, all_vals_num #all stacked values
		
def events2csv(dirpath = "./msrn/experiment"):
	experiment_paths=glob.glob(os.path.join(dirpath,'*/'))
	for experiment_path in experiment_paths:
		print("Parsing tfevents from "+experiment_path)
		if len(glob.glob(os.path.join(experiment_path,'events.out.tfevents*'))):
			experiment_name=os.path.basename(os.path.dirname(experiment_path))
			all_vals = event2arrays(experiment_path)
			np.savetxt(
		        os.path.join(experiment_path, "events.csv"),
		        np.asarray(all_vals),
		        delimiter=",",
		        fmt='%s'
		    )
		else:
			print(experiment_path + " has no tfevents")

def events2csv_stack(dirpath = "./msrn/experiment", plot=True):
	experiment_paths=glob.glob(os.path.join(dirpath,'*/'))
	stacked_vals = []
	stacked_vals_num = []
	stacked_metrics = []
	empty_token=0
	# read tfevents to a list
	for experiment_path in experiment_paths:
		print("Parsing tfevents from "+experiment_path)
		if len(glob.glob(os.path.join(experiment_path,'events.out.tfevents*'))):
			experiment_name=os.path.basename(os.path.dirname(experiment_path))
			all_vals,all_vals_num = event2arrays(experiment_path)
			all_vals_dict = {}
			all_vals_num_dict = {}
			for m,metric in enumerate(all_vals):
				metric_name = metric[0].replace("/","_")
				all_vals_dict[metric_name]=metric[1:]
				all_vals_num_dict[metric_name]=all_vals_num[m]
				stacked_metrics.append(metric_name)
			stacked_vals.append(all_vals_dict)
			stacked_vals_num.append(all_vals_num_dict)
		else:
			print(experiment_path + " has no tfevents")
	stacked_metrics=list(set(stacked_metrics))
	for metric in stacked_metrics:
		all_rows = []
		all_rows_data = []
		all_rows_tags = []
		# stack experiment values per metric
		for i,all_vals in enumerate(stacked_vals):
			experiment_path = experiment_paths[i]
			if len(glob.glob(os.path.join(experiment_path,'events.out.tfevents*'))):
				experiment_name=os.path.basename(os.path.dirname(experiment_path))
				if metric in all_vals.keys():
					metric_row=all_vals[metric]
					all_rows_tags.append(experiment_name)
					metric_row.insert(0,experiment_name)
					all_rows.append(metric_row)
					all_rows_data.append(stacked_vals_num[i][metric]) 
		if not len(all_rows):
			break
		# add empty values to have same cols for each metric (some experiments have more iters/epochs)
		maxlen=len(max(all_rows,key=len))
		for r,row in enumerate(all_rows):
			lendiff=maxlen-len(all_rows[r])
			if lendiff>0:
				for col in range(lendiff):
					all_rows[r].append(empty_token)
		print("Generating "+os.path.join(dirpath, metric+".csv"))
		np.savetxt(
	        os.path.join(dirpath, metric+".csv"),
	        np.asarray(all_rows),
	        delimiter=",",
	        fmt='%s'
	    )
		if plot:
			print("Plotting "+os.path.join(dirpath, metric+".png"))
			plot_1d(all_rows_data, metric, dirpath, ["iter", metric], all_rows_tags, (10,10))
if __name__ == "__main__":
    # events2csv(dirpath = "./msrn/experiment")
    events2csv_stack(dirpath = "./msrn/experiment", plot=True)
