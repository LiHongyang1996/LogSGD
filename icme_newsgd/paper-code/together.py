import subprocess
import threading
import sys
import time


def print_output(process, name):
    """
    读取并打印子进程的输出。
    """
    try:
        # 逐行读取 stdout
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name} STDOUT] {line.strip()}")
        process.stdout.close()
    except Exception as e:
        print(f"读取 {name} STDOUT 时发生异常: {e}", file=sys.stderr)

    try:
        # 逐行读取 stderr
        for line in iter(process.stderr.readline, ''):
            if line:
                print(f"[{name} STDERR] {line.strip()}", file=sys.stderr)
        process.stderr.close()
    except Exception as e:
        print(f"读取 {name} STDERR 时发生异常: {e}", file=sys.stderr)


def terminate_process(process, name):
    """
    终止子进程。
    """
    if process.poll() is None:  # 进程仍在运行
        print(f"终止 {name} 进程...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print(f"{name} 进程已终止。")
        except subprocess.TimeoutExpired:
            print(f"{name} 进程未能在5秒内终止，强制杀死...", file=sys.stderr)
            process.kill()
            print(f"{name} 进程已被强制杀死。", file=sys.stderr)


def run_scripts():
    try:
        # 启动第一个脚本，并捕获其输出
        process1 = subprocess.Popen(
            ['python', '-u', '/code/newsgd/paper-code/multi_1.py', 'RankKReducer', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 启动第二个脚本，并捕获其输出
        process2 = subprocess.Popen(
            ['python', '-u', '/code/newsgd/paper-code/multi_2.py', 'RankKReducer', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("已启动 multi_1.py 和 multi_2.py。按 Ctrl+C 以中断并终止子进程。")

        # 创建并启动线程来实时打印两个脚本的输出
        thread1 = threading.Thread(target=print_output, args=(process1, 'multi_1.py'), daemon=True)
        thread2 = threading.Thread(target=print_output, args=(process2, 'multi_2.py'), daemon=True)
        thread1.start()
        thread2.start()

        # 主线程持续监控两个子进程
        while True:
            # 检查子进程是否已经终止
            ret1 = process1.poll()
            ret2 = process2.poll()

            if ret1 is not None:
                # process1 已经终止
                print(f"检测到 multi_1.py 已终止（返回码: {ret1}）。正在终止 multi_2.py...")
                terminate_process(process2, 'multi_2.py')
                break

            if ret2 is not None:
                # process2 已经终止
                print(f"检测到 multi_2.py 已终止（返回码: {ret2}）。正在终止 multi_1.py...")
                terminate_process(process1, 'multi_1.py')
                break

            time.sleep(0.5)  # 防止CPU占用过高

    except KeyboardInterrupt:
        print("\n检测到中断信号，正在终止子进程...")
        terminate_process(process1, 'multi_1.py')
        terminate_process(process2, 'multi_2.py')
        sys.exit(1)
    except Exception as e:
        print(f"运行脚本时发生异常: {e}", file=sys.stderr)
        terminate_process(process1, 'multi_1.py')
        terminate_process(process2, 'multi_2.py')
        sys.exit(1)
    finally:
        # 确保所有子进程都已终止
        if 'process1' in locals() and process1.poll() is None:
            terminate_process(process1, 'multi_1.py')
        if 'process2' in locals() and process2.poll() is None:
            terminate_process(process2, 'multi_2.py')

        # 等待输出线程完成
        if 'thread1' in locals() and thread1.is_alive():
            thread1.join()
        if 'thread2' in locals() and thread2.is_alive():
            thread2.join()


def main():
    run_scripts()


if __name__ == "__main__":
    main()
