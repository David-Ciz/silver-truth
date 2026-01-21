
from silver_truth.data_processing.compression import compress_tifs_logic
import silver_truth.fusion.fusion as fusion
#from silver_truth.data_processing.label_synchronizer import verify_dataset_synchronization_logic
from silver_truth.ensemble.datasets import benchmark_EnsembleDataset, Version
import silver_truth.ensemble.external as ext
import silver_truth.ensemble.ensemble as ensemble
import silver_truth.ensemble.utils as utils
import os
from silver_truth.data_processing.utils.parquet_utils import add_split_type, same_splits
from silver_truth.ensemble.models import ModelType
from silver_truth.evaluation.final_evaluation import evaluate_strategies
#import silver_truth.qa.preprocessing as  qa_pp
from silver_truth.qa.preprocessing import create_qa_dataset
from silver_truth.qa.evaluation import integrate_results
from silver_truth.qa.result_conversion import excel2csv
import silver_truth.qa.preprocessing as  qa_pp


# ----- Workflow ----- #

def build_qa_databank(build_opt, original_dataset_dir="data/dataframes", qa_parquet_dir="data/ensemble_data/qa"):
    ##### 1) generate QA parquet
    original_dataset_path = os.path.join(original_dataset_dir, f"{build_opt["name"]}_dataset_dataframe.parquet")
    qa_output_path = os.path.join(qa_parquet_dir, f"qa_{build_opt["name"]}")
    qa_parquet_path = os.path.join(qa_parquet_dir, f"qa_{build_opt["name"]}.parquet")

    # build required QA databanks
    qa_pp.create_qa_dataset(
            original_dataset_path,
            qa_output_path,
            qa_parquet_path,  
            crop_size=build_opt["crop_size"]
        )
    # compress images to save space
    ext.compress_images(qa_output_path)

    # TODO: this should be done with original dataset, before generating QA parquet!
    # find largest cell in order to know the crop_size.
    # if bigger than crop size, recall ensemble.build_databanks() with appropriate crop size.
    max_size, image_path = utils.find_largest_gt_cell_size(qa_parquet_path)
    print(f"\nDataset {build_opt["name"]}, max cell size: {max_size} px. Path: {image_path}\n")


    ##### 2) add splits to QA parquet
    ds_qa_split = add_split_type(qa_parquet_path, build_opt)
    return ds_qa_split


def integrate_qa_results(build_opt_list, qa_parquet_dir="data/ensemble_data/qa"):
    ##### 4) integrate results into a parquet
    #qa_results_list = [
    #    excel2csv(os.path.join(qa_parquet_dir, f"{build_opt["name"]}_QA-a1.xlsx")),
    #    os.path.join(qa_parquet_dir, f"{build_opt["name"]}_QA-b1.parquet")
    #]
    qa_results_dict = {}
    for build_opt in build_opt_list:
        if build_opt["qa"] is not None:
            key = os.path.join(qa_parquet_dir, f"qa_{build_opt["name"]}_split{utils.get_splits_name(build_opt)}.parquet")
            value = os.path.join(qa_parquet_dir, f"{build_opt["name"]}_{build_opt["qa"]}.xlsx")
            if key in qa_results_dict:
                if value not in qa_results_dict[key]:
                    qa_results_dict[key].append(value)
            else:
                qa_results_dict[key] = [value]

    for qa_split in qa_results_dict:
        converted_qa_results = []
        for qa_results in qa_results_dict[qa_split]:
            converted_qa_results.append(excel2csv(qa_results))
        integrate_results(qa_split, converted_qa_results)
                      

def build_ensemble_databanks(build_opt_list, qa_parquet_dir="data/ensemble_data/qa"):
    ##### 5) build Ensemble databanks
    ensemble_databanks = []
    for build_opt in build_opt_list:
        qa_parquet_path = os.path.join(qa_parquet_dir, f"qa_{build_opt["name"]}_split{utils.get_splits_name(build_opt)}_res.parquet")
        ensemble_databanks.append(ensemble.build_databank(build_opt, qa_parquet_path))
 
    return ensemble_databanks


def train_model(ensemble_databank_path):
    ##### 6) train models
    ds_ensemble_split = f"data/ensemble_data/databanks/ensemble_{ds_create_opt["version"].name}_{ds_create_opt["name"]}_split{ds_create_opt["split_seed"]}.parquet"

    # train models
    experiment_name = f"{ds_create_opt["name"]}_exp1"
    databank_name = os.path.basename(ds_ensemble_split)[len("ensemble_"):-len(".parquet")]
    train_parquet_path = f"{os.path.join(os.getcwd(), ds_ensemble_split)}"
    run_sequence = [
        {"model_type": ModelType.Unet, "max_epochs": 2, "databank": None}, 
        #{"model_type": ModelType.UnetPlusPlus, "max_epochs": 100, "qa": None}
    ]
    ensemble.run_experiment(experiment_name, databank_name, train_parquet_path, run_sequence)


def evaluate_models(models_paths, build_opt_list):
    for model_path in models_paths:
        for build_opt in build_opt_list:
            databanks_path = os.path.join(utils.DATABANKS_DIR,f"{utils.get_databank_name(build_opt)}.parquet")
            ensemble.generate_evaluation(model_path, databanks_path, "all")


def do_fusion():
    qa_fusion_output_path = "data/qa_data/qa_images_BF-C2DL-HSC"
    qa_fusion_parquet_path = "data/qa_data/qa_images_BF-C2DL-HSC.parquet"
    #qa_pp.create_qa_dataset("data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet", qa_fusion_output_path, qa_fusion_parquet_path, crop=False)
    #compress_tifs_logic(qa_fusion_output_path, True, False, False)

    #compress_tifs_logic("data/fused/BF-C2DL-HSC", True, False, False)

    original_parquet = "data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet"
    fusion_results_path = "data/fused/BF-C2DL-HSC_majority_flat"
    fusion_output_parquet_path = "data/fused/BF-C2DL-HSC_majority_flat.parquet"
    fusion.build_results_databank(original_parquet, fusion_results_path, fusion_output_parquet_path)

    #add_split_type(fusion_output_parquet_path, 42, [0.7,0.15,0.15])

    fusion_results_parquet = "data/fused/BF-C2DL-HSC_majority_flat_split42.parquet"
    #fusion.generate_evaluation(fusion_results_parquet)

    results_example = [{"strategy": "majority_flat_BF-C2DL-HSC_testset", "file": "data/fused/eval_BF-C2DL-HSC_majority_flat_split42_testset.parquet"}]
    #evaluate_strategies("evaluation_test1", results_example)
    

#"""
build_opt_list = [
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": None},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.50},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.55},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.60},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.65},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.70},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.75},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.80},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.85},
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": "QA-eb7-1", "qa_threshold": 0.90},
    #{"name": "BF-C2DL-MuSC", "version": Version.C1, "crop_size": 512, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": None},
]
"""

build_opt_list = [
    {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15], 
     "qa": ["QA-b1", "QA-eb7-1"], "qa_threshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]},
]
"""
#qa_parquet_path = build_qa_databank(build_opt_list[0])

##### 3) get results from QA

##### 4)
#integrate_qa_results(build_opt_list)

## OPTIONAL: build analysis databanks in order to better visualize the data
#ensemble.build_analysis_databanks(build_opt_list[0]["name"], qa_parquet_path, 'all')
#ensemble_databanks = build_ensemble_databanks(build_opt_list)

#train_model()

models_paths = [
    "data/ensemble_data/results/checkpoints/C1_ds1-42-7015_QA--/M1--.ckpt",
    "data/ensemble_data/results/checkpoints/C1_ds1-42-7015_QA--/M2--.ckpt"
]
#evaluate_models(models_paths, build_opt_list)

a = 0

# ----- ######## ----- #






#p_path = "data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet"
#df = ext.load_parquet(p_path)

#for filename in os.listdir("data/fused/BF-C2DL-HSC/02"):
#    os.rename("data/fused/BF-C2DL-HSC/02/"+filename, "data/fused/BF-C2DL-HSC/02/"+"fused_"+filename.split("fused_")[-1])

####### FUSION #######
"""
qa_fusion_output_path = "data/qa_data/qa_images_BF-C2DL-HSC"
qa_fusion_parquet_path = "data/qa_data/qa_images_BF-C2DL-HSC.parquet"
#qa_pp.create_qa_dataset("data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet", qa_fusion_output_path, qa_fusion_parquet_path, crop=False)
#compress_tifs_logic(qa_fusion_output_path, True, False, False)

#compress_tifs_logic("data/fused/BF-C2DL-HSC", True, False, False)

original_parquet = "data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet"
fusion_results_path = "data/fused/BF-C2DL-HSC_majority_flat"
fusion_output_parquet_path = "data/fused/BF-C2DL-HSC_majority_flat.parquet"
#fusion.build_results_databank(original_parquet, fusion_results_path, fusion_output_parquet_path)

#add_split_type(fusion_output_parquet_path, 42, [0.7,0.15,0.15])

fusion_results_parquet = "data/fused/BF-C2DL-HSC_majority_flat_split42.parquet"
#fusion.generate_evaluation(fusion_results_parquet)

results_example = [{"strategy": "majority_flat_BF-C2DL-HSC_testset", "file": "data/fused/eval_BF-C2DL-HSC_majority_flat_split42_testset.parquet"}]
#evaluate_strategies("evaluation_test1", results_example)
"""

#do_fusion()

####### ------ #######


dataset_name = "BF-C2DL-MuSC"


"""
verify_dataset_synchronization_logic(
        "data/synchronized_data/BF-C2DL-HSC", 
        "data/synchronized_data/BF-C2DL-HSC", 
        "data/synchronized_data/BF-C2DL-HSC"
    )
compress_tifs_logic("data/synchronized_data/BF-C2DL-HSC", True, False, False)
"""




"""
compress_tifs_logic(qa_output_path, True, False, False)

"""


#ensemble.build_required_datasets(Version.V1)

#ensemble_dataset_v001_parquet_path = "data/ensemble_data/datasets/v1.00/ensemble_dataset_v1.00.parquet"
#benchmark_EnsembleDataset(ensemble_dataset_v001_parquet_path)

from matplotlib import pyplot as mp
import numpy as np

"""
def gaussian(x, mu, sig):
    return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

x_values = np.linspace(-3, 3, 120)
#for mu, sig in [(-1, 1), (0, 2), (2, 3)]:
for mu, sig in [(-1.25, 0.4)]:
    mp.plot(x_values, gaussian(x_values-2, mu, sig))

mp.show()
"""

"""
p1 = add_split_type('data/ensemble_data/qa/qa_BF-C2DL-HSC.parquet', 42, [0.7,0.15,0.15])
p2 = add_split_type('data/ensemble_data/datasets/v1.00/ensemble_dataset_v1.00.parquet', 42, [0.7,0.15,0.15])

same_splits_result = same_splits(p1, p2)
print("Same splits: ",same_splits_result)
"""



dataset_path = "data/ensemble_data/datasets/v1.00/ensemble_BF-C2DL-HSC_split42.parquet"
#model_path = "data/ensemble_data/results/checkpoints/model_MSE/lightning_logs/version_0/checkpoints/epoch=41-step=798.ckpt"
model_path = "data/ensemble_data/results/checkpoints/UnetPlusPlus.ckpt"
#ensemble.generate_evaluation(model_path,dataset_path, "test")

#experiment_name = "test_ensemble_exp2"
experiment_name = "text_ensemble_exp1"
parquet_path = f"{os.path.join(os.getcwd(), dataset_path)}"
run_sequence = [
    {"model_type": ModelType.Unet, "max_epochs": 100}, 
    {"model_type": ModelType.UnetPlusPlus, "max_epochs": 100}
]
#ensemble.run_experiment(experiment_name, parquet_path, run_sequence)


results_example = [#{"strategy": "Unet_BF-C2DL-HSC_split42_testset", "file": "data/ensemble_data/results/checkpoints/eval_Unet_BF-C2DL-HSC_split42_testset.parquet"},
                   {"strategy": "UnetPlusPlus_BF-C2DL-HSC_split42_testset", "file": "data/ensemble_data/results/checkpoints/eval_UnetPlusPlus_BF-C2DL-HSC_split42_testset.parquet"}
                   ]

results_example = [{"strategy": "majority_flat_BF-C2DL-HSC_split42_testset", "file": "data/fused/eval_BF-C2DL-HSC_majority_flat_split42_testset.parquet"}]
#evaluate_strategies("evaluation_test1", results_example)