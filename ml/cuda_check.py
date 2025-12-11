import torch


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device)}")

        x = torch.randn(2048, 2048, device=device)
        y = x @ x
        print(f"CUDA test succeeded, tensor mean = {y.mean().item():.6f}")
    else:
        print("CUDA not available – check NVIDIA driver and PyTorch installation.")


if __name__ == "__main__":
    main()
