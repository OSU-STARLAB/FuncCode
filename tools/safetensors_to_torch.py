import argparse
import torch
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint")
    args = parser.parse_args()

    state = load_file(args.input)
    torch.save(state, args.output)
    print(f"Converted {args.input} -> {args.output}")

if __name__ == "__main__":
    main()
