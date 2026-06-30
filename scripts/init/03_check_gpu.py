import sys


def main():
    try:
        import torch
    except ImportError as error:
        print(f"[error] torch absent: {error}")
        sys.exit(1)

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda built: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"device count: {torch.cuda.device_count()}")
        print(f"device 0: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
