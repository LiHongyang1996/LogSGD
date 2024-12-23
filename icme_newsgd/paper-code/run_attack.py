import subprocess

# 完整的42条命令
commands = [
    "python gia.py ExactReducer nonquantized low mnist",
    "python gia.py SignAndNormReducer nonquantized low mnist",
    "python gia.py RankKReducer nonquantized low mnist",
    "python gia.py RankKReducer nonquantized mid mnist",
    "python gia.py RankKReducer nonquantized high mnist",
    "python gia.py RankKReducer quantized low mnist",
    "python gia.py RankKReducer quantized mid mnist",
    "python gia.py RankKReducer quantized high mnist",
    "python gia.py TopKReducer nonquantized low mnist",
    "python gia.py TopKReducer nonquantized mid mnist",
    "python gia.py TopKReducer nonquantized high mnist",
    "python gia.py TopKReducer quantized low mnist",
    "python gia.py TopKReducer quantized mid mnist",
    "python gia.py TopKReducer quantized high mnist",
    "python gia.py ExactReducer nonquantized low cifar10",
    "python gia.py SignAndNormReducer nonquantized low cifar10",
    "python gia.py RankKReducer nonquantized low cifar10",
    "python gia.py RankKReducer nonquantized mid cifar10",
    "python gia.py RankKReducer nonquantized high cifar10",
    "python gia.py RankKReducer quantized low cifar10",
    "python gia.py RankKReducer quantized mid cifar10",
    "python gia.py RankKReducer quantized high cifar10",
    "python gia.py TopKReducer nonquantized low cifar10",
    "python gia.py TopKReducer nonquantized mid cifar10",
    "python gia.py TopKReducer nonquantized high cifar10",
    "python gia.py TopKReducer quantized low cifar10",
    "python gia.py TopKReducer quantized mid cifar10",
    "python gia.py TopKReducer quantized high cifar10",
    "python gia.py ExactReducer nonquantized low cifar100",
    "python gia.py SignAndNormReducer nonquantized low cifar100",
    "python gia.py RankKReducer nonquantized low cifar100",
    "python gia.py RankKReducer nonquantized mid cifar100",
    "python gia.py RankKReducer nonquantized high cifar100",
    "python gia.py RankKReducer quantized low cifar100",
    "python gia.py RankKReducer quantized mid cifar100",
    "python gia.py RankKReducer quantized high cifar100",
    "python gia.py TopKReducer nonquantized low cifar100",
    "python gia.py TopKReducer nonquantized mid cifar100",
    "python gia.py TopKReducer nonquantized high cifar100",
    "python gia.py TopKReducer quantized low cifar100",
    "python gia.py TopKReducer quantized mid cifar100",
    "python gia.py TopKReducer quantized high cifar100",
]

# 顺序执行命令
for cmd in commands:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)  # 阻塞式运行命令
    print(f"Completed: {cmd}")