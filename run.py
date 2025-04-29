import os

def _execute(args, use_cuda=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda)
    script = "python train.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)

def run(use_cuda=0):
    """Args for EVOS(stepwise scheduler) with only one image"""
    args = [
        "--name",
        "1x64_out1.0_omega10.0",
        "--config",
        "1x64",
        "--model",
        "WT",
        "--dataset",
        "cifar10_feat",
        "--device",
        "cuda:0",

        "--output_scale",
        "1.0",

    ]
    _execute(args, use_cuda)

if __name__ == "__main__":
    run(2)

