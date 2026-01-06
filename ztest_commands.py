
from src.data_processing.compression import compress_tifs_logic
import src.fusion.fusion as fusion
#from src.data_processing.label_synchronizer import verify_dataset_synchronization_logic
from src.ensemble.datasets import benchmark_EnsembleDataset, Version
import src.ensemble.external as ext
import src.ensemble.ensemble as ensemble
import src.ensemble.utils as utils
import os
from src.data_processing.utils.parquet_utils import add_split_type, same_splits
from src.ensemble.models import ModelType
from src.evaluation.final_evaluation import evaluate_strategies
#import src.qa.preprocessing as  qa_pp
from src.qa.preprocessing import create_qa_dataset

# ----- Workflow ----- #

#ds_create_opt = {"name": "BF-C2DL-HSC", "version": Version.C1, "crop_size": 64, "split_seed": 42, "split_sets": [0.7,0.15,0.15]}
ds_create_opt = {"name": "BF-C2DL-MuSC", "version": Version.C1, "crop_size": 512, "split_seed": 42, "split_sets": [0.7,0.15,0.15]}
"""
# build QA and Ensemble databanks
original_dataset_path = f"data/dataframes/{ds_create_opt["name"]}_dataset_dataframe.parquet"
ensemble_parquet_path, qa_parquet_path = ensemble.build_databanks(
                                                    original_dataset_path, 
                                                    ds_create_opt["name"], 
                                                    ds_create_opt["version"], 
                                                    ds_create_opt["crop_size"]
                                                )

# find largest cell in order to know the crop_size.
# if bigger than crop size, recall ensemble.build_databanks() with appropriate crop size.
max_size, image_path = utils.find_largest_gt_cell_size(qa_parquet_path)
print(f"\nDataset {ds_create_opt["name"]}, max cell size: {max_size} px. Path: {image_path}\n")

# build analysis databanks in order to better visualize the data
ensemble.build_analysis_databanks(ds_create_opt["name"], 'all')

# add split to databanks parquets
ds_qa_split = add_split_type(qa_parquet_path, ds_create_opt["split_seed"], ds_create_opt["split_sets"])
ds_ensemble_split = add_split_type(ensemble_parquet_path, ds_create_opt["split_seed"], ds_create_opt["split_sets"])

# confirm that the splits are the same
same_splits_result = same_splits(ds_qa_split, ds_ensemble_split)
print("Same splits: ",same_splits_result)
assert(same_splits_result)
#"""
ds_ensemble_split = f"data/ensemble_data/datasets/ensemble_{ds_create_opt["version"].name}_{ds_create_opt["name"]}_split{ds_create_opt["split_seed"]}.parquet"

# train models
experiment_name = f"{ds_create_opt["name"]}_exp1"
train_parquet_path = f"{os.path.join(os.getcwd(), ds_ensemble_split)}"
run_sequence = [
    {"model_type": ModelType.Unet, "max_epochs": 100}, 
    {"model_type": ModelType.UnetPlusPlus, "max_epochs": 100}
]
ensemble.run_experiment(experiment_name, train_parquet_path, run_sequence)

a = 0

#...

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
evaluate_strategies("evaluation_test1", results_example)