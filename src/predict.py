import argparse


def greeting(name: str) -> str:
    return f"Hello, {name}!"


def main():
    parser = argparse.ArgumentParser(
        description="Print a greeting message with the provided name"
    )
    
    parser.add_argument(
        "name",
        type=str,
        help="Your name to be included in the greeting"
    )    
    
    args = parser.parse_args()
    print(greeting(args.name))


if __name__ == "__main__":
    main()