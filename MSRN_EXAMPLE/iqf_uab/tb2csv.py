from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from iq_tool_box.quality_metrics.benchmark import plot_1d, get_topk, formatvals
import numpy as np
import glob
import os

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
            try:
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
            except:
                print("ERROR when loading: "+ experiment_path)
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
        # generate all rows data csv
        print("Generating "+os.path.join(dirpath, metric+".csv"))
        np.savetxt(
	        os.path.join(dirpath, metric+".csv"),
	        np.asarray(all_rows),
	        delimiter=",",
	        fmt='%s'
	    )
        print("Generating Min Max Tables"+os.path.join(dirpath, metric+".csv"))
    	# get best values for all rows
        all_rows_lastvals=[row[-60:len(row)] for row in all_rows_data] # 60 iter corresponds to 10 epoch, namely (1200 iter / 200 epoch) * 10
        lastvals_mean=[np.mean(row) for row in all_rows_lastvals]
        if "FID" in metric or "LOSS" in metric:
            top_k_idx=get_topk(all_rows_data,11,False)
            bestvals=[np.min(row) for row in all_rows_lastvals] # axis=1
        else:
            top_k_idx=get_topk(all_rows_data,11,True)
            bestvals=[np.max(row) for row in all_rows_lastvals]
        all_rows_bestvals=[list(val) for val in zip(all_rows_tags,formatvals(bestvals,True))]
        # filter top K (11) values to allow visible plotting (discard low value experiments)
        all_rows_data_top=[all_rows_lastvals[idx] for idx in top_k_idx]
        all_rows_tags_top=[all_rows_tags[idx] for idx in top_k_idx]
        bestvals_top=[bestvals[idx] for idx in top_k_idx]
        # generate csv with best values for each experiment
        np.savetxt(
            os.path.join(dirpath, "best_"+metric+".csv"),
            np.asarray(all_rows_bestvals),
            delimiter=",",
            fmt='%s'
        )
        # generate plots
        print("Plotting "+os.path.join(dirpath, metric+".png"))
        if plot:
        	plot_1d(all_rows_data_top, "box_"+metric, dirpath, ["iter", metric], all_rows_tags_top, (10,10), "boxplot")
        	plot_1d(bestvals_top, "bar_"+metric, dirpath, ["iter", metric], all_rows_tags_top, (10,10), "bar")
        	plot_1d(all_rows_data_top, metric, dirpath, ["iter", metric], all_rows_tags_top, (20,20), "plot")
if __name__ == "__main__":
    # events2csv(dirpath = "./msrn/experiment")
    events2csv_stack(dirpath = "./msrn/experiment_crops512_whole", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops256_whole", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops128_whole", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops64_whole", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops32_whole", plot=True)
    '''
    events2csv_stack(dirpath = "./msrn/experiment_crops512_short", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops256_short", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops128_short", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops64_short", plot=True)
    events2csv_stack(dirpath = "./msrn/experiment_crops32_short", plot=True)
    '''