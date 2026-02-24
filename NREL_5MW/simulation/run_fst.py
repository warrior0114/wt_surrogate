import subprocess
import os


def run_fst(fst_file_road):
    print("=====" + "run fst" + "=====")
    # 定义exe路径和待处理文件路径
    exe_path = "fst_and_results_file/openfast_x64.exe"  # 替换为实际的exe路径
    file_to_process = fst_file_road          # 替换为实际的文件路径

    try:
        # 使用subprocess运行命令
        result = subprocess.run(
            [exe_path, file_to_process],
            check=True,        # 检查返回码，非零时抛出异常
            capture_output=True, # 可选：捕获输出
            text=True           # 可选：以文本形式返回输出
        )
        print("successful run fst")
        # print('\n')
    except subprocess.CalledProcessError as e:
        print(f"ERROR_1{e.stderr}") # 错误：程序返回非零状态码。详细信息
        print('\n')
    except FileNotFoundError:
        print("ERROR_2") # 错误：找不到指定的exe文件。请检查路径是否正确。
        print('\n')
    except Exception as e:
        print(f"NOKOWN{str(e)}") # 未知错误 .

