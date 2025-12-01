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

    return parser.parse_args()

    parser.add 

