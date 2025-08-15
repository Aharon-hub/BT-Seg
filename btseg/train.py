import argparse, os, pathlib, sys, yaml, torch
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nn_transunet.default_configuration import get_default_configuration

def cli():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--network", default="3d_fullres")
    p.add_argument("--network_trainer", default="nnUNetTrainerV2_HD95")
    p.add_argument("--task", default="Task180_BraTS19")
    p.add_argument("--fold", default="0")
    p.add_argument("--resume", default="")
    p.add_argument("-p", default=default_plans_identifier)
    p.add_argument("--validation_only", action="store_true")
    p.add_argument("--npz", action="store_true")
    p.add_argument("--valbest", action="store_true")
    p.add_argument("--vallatest", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--model", default="Generic_TransUNet_max_ppbp")
    p.add_argument("--config", default="")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_num_epochs", type=int, default=1000)
    p.add_argument("--initial_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-5)
    p.add_argument("--lrschedule", default="poly")
    p.add_argument("--model_params", type=dict, default={})
    args, _ = p.parse_known_args()
    if isinstance(args.fold, str) and args.fold.isdigit():
        args.fold = int(args.fold)
    return args

def _ensure_env():
    base = pathlib.Path(os.environ.get("BTSEG_BASE", pathlib.Path(__file__).resolve()).parents[2])
    os.environ.setdefault("nnUNet_raw_data_base", str(base / "nnUNet_raw_data_base"))
    os.environ.setdefault("nnUNet_preprocessed", str(base / "preprocessed"))
    os.environ.setdefault("RESULTS_FOLDER", str(base / "nnUNet_trained_models"))
    for k in ("nnUNet_raw_data_base","nnUNet_preprocessed","RESULTS_FOLDER"):
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

def build_trainer(args):
    try:
        plans, out_dir, ds_dir, batch_dice, stage, trainer_cls = get_default_configuration(args.network, args.task, args.network_trainer, args.p)
    except KeyError as e:
        if str(e).strip("'") == "nnUNetTrainerV2_HD95":
            from btseg.nn_transunet.trainer.nnUNetTrainerV2_HD95 import nnUNetTrainerV2_HD95
            plans, out_dir, ds_dir, batch_dice, stage, _ = get_default_configuration(args.network, args.task, "nnUNetTrainerV2", args.p)
            trainer_cls = nnUNetTrainerV2_HD95
        else:
            raise
    input_size = args.model_params.get("img_size", [64,160,160])
    trainer = trainer_cls(plans, args.fold, output_folder=out_dir, dataset_directory=ds_dir, batch_dice=batch_dice, stage=stage, unpack_data=True, deterministic=True, fp16=not args.fp32, input_size=input_size, args=args)
    return trainer

def main():
    _ensure_env()
    args = cli()
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    trainer = build_trainer(args)
    trainer.initialize(training=not args.validation_only)
    if args.validation_only:
        if args.valbest: trainer.load_best_checkpoint(train=False)
        elif args.vallatest: trainer.load_latest_checkpoint(train=False)
        else: trainer.load_final_checkpoint(train=False)
        trainer.validate(save_softmax=args.npz, validation_folder_name="val", run_postprocessing_on_folds=True)
        return
    if args.resume in ("auto","local_latest"): trainer.load_latest_checkpoint(train=True)
    elif args.resume and os.path.isfile(args.resume): load_pretrained_weights(trainer.network, args.resume)
    trainer.run_training()

if __name__ == "__main__":
    main()
