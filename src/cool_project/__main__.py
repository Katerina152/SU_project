import traceback 
from cool_project.args_parser import parse_args        
from cool_project.training import build_training_experiment


def main():
    args = parse_args()
    print(f"Parsed arguments: {args}")

    if args.function == "training":
        build_training_experiment(args.config)
    else:
        raise NotImplementedError(f"Function '{args.function}' not supported")

if __name__ == "__main__":
    main()

