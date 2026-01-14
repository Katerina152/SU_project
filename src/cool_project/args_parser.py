import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="ViT training runner")

    parser.add_argument(
        "--function",
        type=str, 
        default="training",
        help="Which function to run (training for now)"

    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help= "path to JSON configuration file"

    )

    parser.add_argument("--trial-index", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")


    return parser.parse_args()



