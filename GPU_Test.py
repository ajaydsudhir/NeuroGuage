import torch
import os

def test_gpu():
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
        print("PyTorch Version:", torch.__version__)
        gpu_properties = torch.cuda.get_device_properties(0)
        print("GPU Memory:", f"{gpu_properties.total_memory / 1e9:.2f} GB")
        print("GPU Multiprocessors:", gpu_properties.multi_processor_count)
        torch.cuda.synchronize()
        print("Running GPU Utilization Test")
        a = torch.randn(10000, 10000, device='cuda')
        b = torch.randn(10000, 10000, device='cuda')
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("Tensor operation completed.")
        os.system("start taskmgr")
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    test_gpu()