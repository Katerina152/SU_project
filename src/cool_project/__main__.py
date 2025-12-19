import traceback 
from cool_project.args_parser import parse_args        
from cool_project.training import build_training_experiment
from cool_project.extract_embeddings import build_embedding_experiment
from cool_project.distillation import run_distillation


def main():
    args = parse_args()
    print(f"Parsed arguments: {args}")

    if args.function == "training":
        build_training_experiment(args.config)
    elif args.function == "extract_embeddings":
        build_embedding_experiment(args.config)
    elif args.function == "distillation":
        run_distillation(args.config)
    else:
        raise NotImplementedError(f"Function '{args.function}' not supported")

if __name__ == "__main__":
    main()

