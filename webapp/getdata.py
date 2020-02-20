from torchvision import datasets, transforms

if __name__ == "__main__":
    x = datasets.MNIST("data", download=True, train=False)
