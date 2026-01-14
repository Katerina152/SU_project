import traceback 
from cool_project.args_parser import parse_args        
from cool_project.training import build_training_experiment
from cool_project.extract_embeddings import build_embedding_experiment
from cool_project.distillation import run_distillation
from cool_project.training import (
    load_config,
    build_training_experiment,
    run_trial_by_index,
    run_hparam_sweep,
)


def main():
    args = parse_args()
    print(f"Parsed arguments: {args}")

    if args.function == "training":
        # 1) SLURM array: run exactly one combo
        if args.trial_index is not None:
            run_trial_by_index(args.config, args.trial_index)
            return

        # 2) Otherwise: decide sweep vs single run
        cfg = load_config(args.config)
        do_sweep = args.sweep or cfg.get("tune", {}).get("enabled", False)

        if do_sweep:
            run_hparam_sweep(args.config)  # multi-trial in one job
        else:
            build_training_experiment(args.config, run_tag=None, run_test=True)

    elif args.function == "extract_embeddings":
        build_embedding_experiment(args.config)
    elif args.function == "distillation":
        run_distillation(args.config)
    else:
        raise NotImplementedError(f"Function '{args.function}' not supported")

if __name__ == "__main__":
    main()

