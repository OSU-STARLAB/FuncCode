import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(obj, dict):
        print("Top-level keys:")
        for k in obj.keys():
            print(" ", k)
        if "metadata" in obj:
            print("\nMetadata:")
            print(obj["metadata"])

if __name__ == "__main__":
    main()
