import subprocess
import sys

def main():
    # 从命令行获取参数
    if len(sys.argv) < 5:
        print("Usage: python run_all.py <arg1> <arg2> <arg3> <arg4>")
        sys.exit(1)

    # 提取参数
    arg1, arg2, arg3, arg4 = sys.argv[1:5]

    # 创建命令列表
    commands = [
        f"python multi.py {arg1} {arg2} {arg3} {arg4} {i}" for i in range(8)
    ]

    # 启动子进程
    processes = []
    for cmd in commands:
        print(f"Starting: {cmd}")
        processes.append(subprocess.Popen(cmd, shell=True))

    # 等待所有子进程完成
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()