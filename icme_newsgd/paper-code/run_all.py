import subprocess

# 完整的42条命令
commands = [
    "python all.py ExactReducer nonquantized low mnist",
    "python all.py SignAndNormReducer nonquantized low mnist",
    "python all.py RankKReducer nonquantized low mnist",
    "python all.py RankKReducer nonquantized mid mnist",
    "python all.py RankKReducer nonquantized high mnist",
    "python all.py RankKReducer quantized low mnist",
    "python all.py RankKReducer quantized mid mnist",
    "python all.py RankKReducer quantized high mnist",
    "python all.py TopKReducer nonquantized low mnist",
    "python all.py TopKReducer nonquantized mid mnist",
    "python all.py TopKReducer nonquantized high mnist",
    "python all.py TopKReducer quantized low mnist",
    "python all.py TopKReducer quantized mid mnist",
    "python all.py TopKReducer quantized high mnist",
    "python all.py ExactReducer nonquantized low cifar10",
    "python all.py SignAndNormReducer nonquantized low cifar10",
    "python all.py RankKReducer nonquantized low cifar10",
    "python all.py RankKReducer nonquantized mid cifar10",
    "python all.py RankKReducer nonquantized high cifar10",
    "python all.py RankKReducer quantized low cifar10",
    "python all.py RankKReducer quantized mid cifar10",
    "python all.py RankKReducer quantized high cifar10",
    "python all.py TopKReducer nonquantized low cifar10",
    "python all.py TopKReducer nonquantized mid cifar10",
    "python all.py TopKReducer nonquantized high cifar10",
    "python all.py TopKReducer quantized low cifar10",
    "python all.py TopKReducer quantized mid cifar10",
    "python all.py TopKReducer quantized high cifar10",
    "python all.py ExactReducer nonquantized low cifar100",
    "python all.py SignAndNormReducer nonquantized low cifar100",
    "python all.py RankKReducer nonquantized low cifar100",
    "python all.py RankKReducer nonquantized mid cifar100",
    "python all.py RankKReducer nonquantized high cifar100",
    "python all.py RankKReducer quantized low cifar100",
    "python all.py RankKReducer quantized mid cifar100",
    "python all.py RankKReducer quantized high cifar100",
    "python all.py TopKReducer nonquantized low cifar100",
    "python all.py TopKReducer nonquantized mid cifar100",
    "python all.py TopKReducer nonquantized high cifar100",
    "python all.py TopKReducer quantized low cifar100",
    "python all.py TopKReducer quantized mid cifar100",
    "python all.py TopKReducer quantized high cifar100",
]

# 顺序执行命令
for cmd in commands:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)  # 阻塞式运行命令
    print(f"Completed: {cmd}")