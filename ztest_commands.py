import os

import matplotlib.pyplot as plt
# from src.data_processing.label_synchronizer import verify_dataset_synchronization_logic
from silver_truth.ensemble.databanks_builds import Databank_type
from silver_truth.ensemble.datasets import Version
import silver_truth.ensemble.external as ext
import silver_truth.ensemble.ensemble as ensemble
import silver_truth.ensemble.utils as utils
from silver_truth.data_processing.utils.parquet_utils import add_split_type
from silver_truth.ensemble.models import ModelType
# import src.qa.preprocessing as  qa_pp
from silver_truth.qa.evaluation import integrate_results
from silver_truth.qa.result_conversion import excel2csv
import silver_truth.qa.preprocessing as qa_pp


# ----- Workflow ----- #


def build_qa_databank(
    build_opt,
    original_dataset_dir="data/dataframes",
    qa_parquet_dir="data/ensemble_data/qa",
):
    ##### 1) generate QA parquet
    original_dataset_path = os.path.join(
        original_dataset_dir, f"{build_opt['name']}_dataset_dataframe.parquet"
    )
    qa_output_path = os.path.join(qa_parquet_dir, f"qa_{build_opt['name']}")
    qa_parquet_path = os.path.join(qa_parquet_dir, f"qa_{build_opt['name']}.parquet")

    # build required QA databanks
    qa_pp.create_qa_dataset(
        original_dataset_path,
        qa_output_path,
        qa_parquet_path,
        crop="crop_size" in build_opt,
        crop_size=build_opt["crop_size"] if "crop_size" in build_opt else 0,
    )
    # compress images to save space
    ext.compress_images(qa_output_path)

    # TODO: this should be done with original dataset, before generating QA parquet!
    # find largest cell in order to know the crop_size.
    # if bigger than crop size, recall ensemble.build_databanks() with appropriate crop size.
    max_size, image_path = utils.find_largest_gt_cell_size(qa_parquet_path)
    print(
        f"\nDataset {build_opt['name']}, max cell size: {max_size} px. Path: {image_path}\n"
    )

    ##### 2) add splits to QA parquet
    ds_qa_split = add_split_type(qa_parquet_path, build_opt)
    return ds_qa_split


def show_cell_size_hist(build_opt, qa_parquet_dir="data/ensemble_data/qa", num_bins=20):
    hist = utils.get_cell_size_hist(os.path.join(qa_parquet_dir, f"qa_{build_opt['name']}.parquet"), num_bins)
    height_n, height_bins = hist["height_hist"]
    width_n, width_bins = hist["width_hist"]
    #plt.plot(height_bins[:-1], height_n)
    plt.hist(hist["height_raw"], num_bins)
    plt.title(f"{build_opt['name']} cell height histogram")
    plt.show()
    plt.waitforbuttonpress()
    #plt.plot(width_bins[:-1], width_n)
    plt.hist(hist["width_raw"], num_bins)
    plt.title(f"{build_opt['name']} cell width histogram")
    plt.show()
    plt.waitforbuttonpress()


def integrate_qa_results(build_opt_list, qa_parquet_dir="data/ensemble_data/qa"):
    ##### 4) integrate results into a parquet
    # qa_results_list = [
    #    excel2csv(os.path.join(qa_parquet_dir, f"{build_opt['name']}_QA-a1.xlsx")),
    #    os.path.join(qa_parquet_dir, f"{build_opt['name']}_QA-b1.parquet")
    # ]
    qa_results_dict = {}
    for build_opt in build_opt_list:
        if build_opt["qa"] is not None:
            key = os.path.join(
                qa_parquet_dir,
                f"qa_{build_opt['name']}_split{utils.get_splits_name(build_opt)}.parquet",
            )
            value = os.path.join(
                qa_parquet_dir, f"{build_opt['name']}_{build_opt['qa']}.xlsx"
            )
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
        qa_parquet_path = os.path.join(
            qa_parquet_dir,
            f"qa_{build_opt['name']}_split{utils.get_splits_name(build_opt)}_res.parquet",
        )
        qa_parquet_path = "data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet"
        ensemble_databanks.append(ensemble.build_databank(build_opt, qa_parquet_path))

    return ensemble_databanks


def train_model(databank_opt, run_sequence):
    ##### 6) train models
    experiment_name = f"{utils.get_databank_name(databank_opt)}_exp1"
    ensemble.run_experiment__old(experiment_name, run_sequence)


def evaluate_models(models_paths, build_opt_list):
    for model_path in models_paths:
        for build_opt in build_opt_list:
            databanks_path = os.path.join(
                utils.DATABANKS_DIR, f"{utils.get_databank_name(build_opt)}.parquet"
            )
            ensemble.generate_evaluation(model_path, databanks_path, "all")


build_opt_list = [
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": None,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.50,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.55,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.60,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.65,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.70,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.75,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.80,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.85,
    },
    {
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        "split_sets": [0.7, 0.15, 0.15],
        "qa": "QA-eb7-1",
        "qa_threshold": 0.90,
    },
    # {"name": "BF-C2DL-MuSC", "dataset": Version.C1, "crop_size": 512, "split_seed": 42, "split_sets": [0.7,0.15,0.15], "qa": None},
]

build_opt_list = [
    {
        #"name": "BF-C2DL-MuSC",
        "name": "BF-C2DL-HSC",
        "databank": Databank_type.Norm,
        "dataset": Version.C1,
        "crop_size": 64,
        "split_seed": 42,
        #"split_sets": [0.7, 0.15, 0.15],
        "split_sets": [0.8, 0.2, 0.0],
        "qa": None,
    },]

#qa_parquet_path = build_qa_databank(build_opt_list[0])

##### 3) get results from QA

##### 4)
#integrate_qa_results(build_opt_list)


## OPTIONAL: build analysis databanks in order to better visualize the data
#ensemble.build_analysis_databanks(build_opt_list[0]["name"], os.path.join("data/ensemble_data/qa", f"qa_{build_opt_list[0]['name']}.parquet"), 'crop')

## OPTIONAL: check cell size histogram
#show_cell_size_hist(build_opt_list[0])

#ensemble_databanks = build_ensemble_databanks(build_opt_list)

databank_opt = build_opt_list[0]
run_sequence = [
        ####{"model_type": ModelType.Unet_Dynamic, "max_epochs": 2, "databank_opt": databank_opt},
        ####{"model_type": ModelType.Unet_Mult_Input, "max_epochs": 100, "databank_opt": databank_opt},

        #{"model_type": ModelType.UnetPlusPlus, "model_enc": "resnet18", "pretrain":"imagenet", "max_epochs": 100, "databank_opt": databank_opt},
        #{"model_type": ModelType.UnetPlusPlus, "model_enc": "resnet18", "pretrain":"ssl", "max_epochs": 100, "databank_opt": databank_opt},
        #{"model_type": ModelType.UnetPlusPlus, "model_enc": "resnet18", "pretrain":"swsl", "max_epochs": 100, "databank_opt": databank_opt},
        #{"model_type": ModelType.UnetPlusPlus, "model_enc": "resnet34", "pretrain":"imagenet", "max_epochs": 100, "databank_opt": databank_opt},
    ]

MODEL_ENCODER_WEIGHTS = {
    #https://smp.readthedocs.io/en/latest/encoders.html

    "resnet18": ["imagenet", "ssl", "swsl"],        # 11M
    "resnet34": ["imagenet"],                       # 21M
    "resnet50": ["imagenet", "ssl", "swsl"],        # 23M
    "resnet101": ["imagenet"],                      # 42M
    "resnet152": ["imagenet"],                      # 58M
    "resnext50_32x4d": ["imagenet", "ssl", "swsl"], # 22M
    "resnext101_32x4d": ["ssl", "swsl"],            # 42M
    "resnext101_32x8d": ["imagenet", "instagram", "ssl", "swsl"],   # 86M
    "resnext101_32x16d": ["instagram", "ssl", "swsl"],              # 191M
    #"resnext101_32x32d": ["instagram"],             # 466M
    #"resnext101_32x48d": ["instagram"],             # 826M

    "dpn68": ["imagenet"],      # 11M
    "dpn68b": ["imagenet+5k"],  # 11M
    "dpn92": ["imagenet+5k"],   # 34M
    "dpn98": ["imagenet"],      # 58M
    "dpn107": ["imagenet+5k"],  # 84M
    "dpn131": ["imagenet"],     # 76M

    "vgg11": ["imagenet"],      # 9M
    "vgg11_bn": ["imagenet"],   # 9M
    "vgg13": ["imagenet"],      # 9M
    "vgg13_bn": ["imagenet"],   # 9M
    "vgg16": ["imagenet"],      # 14M
    "vgg16_bn": ["imagenet"],   # 14M
    "vgg19": ["imagenet"],      # 20M
    "vgg19_bn": ["imagenet"],   # 20M

    "senet154": ["imagenet"],   # 113M

    "se_resnet50": ["imagenet"],            # 26M
    "se_resnet101": ["imagenet"],           # 47M
    "se_resnet152": ["imagenet"],           # 64M
    "se_resnext50_32x4d": ["imagenet"],     # 25M
    "se_resnext101_32x4d": ["imagenet"],    # 46M

    "densenet121": ["imagenet"],    # 6M
    "densenet169": ["imagenet"],    # 12M
    "densenet201": ["imagenet"],    # 18M
    "densenet161": ["imagenet"],    # 26M

    "inceptionresnetv2": ["imagenet"],   # 54M
    "inceptionv4": ["imagenet"],         # 41M

    "efficientnet-b0": ["imagenet", "advprop"],     # 4M
    "efficientnet-b1": ["imagenet", "advprop"],     # 6M
    "efficientnet-b2": ["imagenet", "advprop"],     # 7M
    "efficientnet-b3": ["imagenet", "advprop"],     # 10M
    "efficientnet-b4": ["imagenet", "advprop"],     # 17M
    "efficientnet-b5": ["imagenet", "advprop"],     # 28M
    "efficientnet-b6": ["imagenet", "advprop"],     # 40M
    "efficientnet-b7": ["imagenet", "advprop"],     # 63M

    "mobilenet_v2": ["imagenet"],   # 2M

    "xception": ["imagenet"],       # 20M

    "timm-efficientnet-b0": ["imagenet", "advprop", "noisy-student"],   # 4M
    "timm-efficientnet-b1": ["imagenet", "advprop", "noisy-student"],   # 6M
    "timm-efficientnet-b2": ["imagenet", "advprop", "noisy-student"],   # 7M
    "timm-efficientnet-b3": ["imagenet", "advprop", "noisy-student"],   # 10M
    "timm-efficientnet-b4": ["imagenet", "advprop", "noisy-student"],   # 17M
    "timm-efficientnet-b5": ["imagenet", "advprop", "noisy-student"],   # 28M
    "timm-efficientnet-b6": ["imagenet", "advprop", "noisy-student"],   # 40M
    "timm-efficientnet-b7": ["imagenet", "advprop", "noisy-student"],   # 63M
    "timm-efficientnet-b8": ["imagenet", "advprop"],                    # 84M
    #"timm-efficientnet-l2": ["noisy-student", "noisy-student-475"],     # 474M

    "timm-tf_efficientnet_lite0": ["imagenet"],     # 3M
    "timm-tf_efficientnet_lite1": ["imagenet"],     # 4M
    "timm-tf_efficientnet_lite2": ["imagenet"],     # 4M
    "timm-tf_efficientnet_lite3": ["imagenet"],     # 6M
    "timm-tf_efficientnet_lite4": ["imagenet"],     # 11M

    "timm-skresnet18": ["imagenet"],        # 11M
    "timm-skresnet34": ["imagenet"],        # 21M
    "timm-skresnext50_32x4d": ["imagenet"], # 23M

    "mit_b0": ["imagenet"],     # 3M
    "mit_b1": ["imagenet"],     # 13M
    "mit_b2": ["imagenet"],     # 24M
    "mit_b3": ["imagenet"],     # 44M
    "mit_b4": ["imagenet"],     # 60M
    "mit_b5": ["imagenet"],     # 81M

    "mobileone_s0": ["imagenet"],   # 4M
    "mobileone_s1": ["imagenet"],   # 3M
    "mobileone_s2": ["imagenet"],   # 5M
    "mobileone_s3": ["imagenet"],   # 8M
    "mobileone_s4": ["imagenet"],   # 12M
}

run_sequence = []
run_models = [ModelType.UnetPlusPlus]
# higher memory models
run_encs = [
    "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x4d", "resnext101_32x8d", "resnext101_32x16d", 
    "dpn107", "dpn131", "vgg19", "vgg19_bn", "senet154", "se_resnet152", 
    "se_resnext101_32x4d", "densenet161", "inceptionresnetv2", "inceptionv4", 
    "efficientnet-b7", "mobilenet_v2", "xception", "timm-efficientnet-b7", "timm-efficientnet-b8", 
    "timm-tf_efficientnet_lite4", "timm-skresnet34", "timm-skresnext50_32x4d", "mit_b5", "mobileone_s4",
    ]

# all models, overrides selection above
run_encs = MODEL_ENCODER_WEIGHTS.keys()
    
# create run sequence dictionary
for run_model in run_models:
    for model_enc in run_encs:
        for model_weight in [None] + MODEL_ENCODER_WEIGHTS[model_enc]:
            run_sequence.append({"model_type": run_model, "model_enc": model_enc, "pretrain":model_weight, "max_epochs": 100, "databank_opt": databank_opt})

#run_sequence = [
#    {"model_type": ModelType.UnetPlusPlus, "model_enc": "resnext101_32x8d", "pretrain":None, "max_epochs": 100, "databank_opt": databank_opt},
#    {"model_type": ModelType.Unet, "model_enc": "resnext101_32x8d", "pretrain":None, "max_epochs": 100, "databank_opt": databank_opt},
#    ]
train_model(databank_opt, run_sequence)

models_paths = [
    "data/ensemble_data/results/checkpoints/C1_ds1-42-7015_QA--/M1--.ckpt",
    "data/ensemble_data/results/checkpoints/C1_ds1-42-7015_QA--/M2--.ckpt",
]
#evaluate_models(models_paths, build_opt_list)


# ----- ######## ----- #


# p_path = "data/dataframes/BF-C2DL-HSC_dataset_dataframe.parquet"
# df = ext.load_parquet(p_path)

# for filename in os.listdir("data/fused/BF-C2DL-HSC/02"):
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


# ensemble.build_required_datasets(Version.V1)

# ensemble_dataset_v001_parquet_path = "data/ensemble_data/datasets/v1.00/ensemble_dataset_v1.00.parquet"
# benchmark_EnsembleDataset(ensemble_dataset_v001_parquet_path)


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
# model_path = "data/ensemble_data/results/checkpoints/model_MSE/lightning_logs/version_0/checkpoints/epoch=41-step=798.ckpt"
model_path = "data/ensemble_data/results/checkpoints/UnetPlusPlus.ckpt"
# ensemble.generate_evaluation(model_path,dataset_path, "test")

# experiment_name = "test_ensemble_exp2"
experiment_name = "text_ensemble_exp1"
parquet_path = f"{os.path.join(os.getcwd(), dataset_path)}"
run_sequence = [
    {"model_type": ModelType.Unet, "max_epochs": 100},
    {"model_type": ModelType.UnetPlusPlus, "max_epochs": 100},
]
# ensemble.run_experiment(experiment_name, parquet_path, run_sequence)


results_example = [  # {"strategy": "Unet_BF-C2DL-HSC_split42_testset", "file": "data/ensemble_data/results/checkpoints/eval_Unet_BF-C2DL-HSC_split42_testset.parquet"},
    {
        "strategy": "UnetPlusPlus_BF-C2DL-HSC_split42_testset",
        "file": "data/ensemble_data/results/checkpoints/eval_UnetPlusPlus_BF-C2DL-HSC_split42_testset.parquet",
    }
]

results_example = [
    {
        "strategy": "majority_flat_BF-C2DL-HSC_split42_testset",
        "file": "data/fused/eval_BF-C2DL-HSC_majority_flat_split42_testset.parquet",
    }
]
# evaluate_strategies("evaluation_test1", results_example)
