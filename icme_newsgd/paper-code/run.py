import subprocess
import sys

def run_commands_sequentially(commands):
    for idx, cmd in enumerate(commands, 1):
        print(f"正在执行第 {idx} 个命令: {cmd}")
        try:
            # 使用subprocess.run执行命令，并等待其完成
            result = subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {cmd}")
            print(f"错误代码: {e.returncode}")
            sys.exit(e.returncode)
        print(f"命令完成: {cmd}\n")
    print("所有命令已按顺序执行完毕。")

if __name__ == "__main__":
    commands = [

        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.0028",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.0056",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.011",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.083",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.14",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 0.27",
        "python /code/powersgd-original_cifar100/paper-code/start.py TopKReducer 1",

        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.0028",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.0056",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.011",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.083",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.14",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 0.27",
        "python /code/powersgd-original/paper-code/start.py TopKReducer 1",

        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 1",
        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 2",
        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 4",
        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 30",
        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 50",
        "python /code/powersgd-original_cifar100/paper-code/start.py RankKReducer 100",

        "python /code/powersgd-original/paper-code/start.py RankKReducer 1",
        "python /code/powersgd-original/paper-code/start.py RankKReducer 2",
        "python /code/powersgd-original/paper-code/start.py RankKReducer 4",
        "python /code/powersgd-original/paper-code/start.py RankKReducer 30",
        "python /code/powersgd-original/paper-code/start.py RankKReducer 50",
        "python /code/powersgd-original/paper-code/start.py RankKReducer 100",
    ]

    run_commands_sequentially(commands)