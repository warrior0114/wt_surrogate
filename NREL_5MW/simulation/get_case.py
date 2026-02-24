import csv
import random

# --- 1. 参数设置 ---

# 固定参数
ANALYSIS_TIME = 630
USABLE_TIME = 600
TURB_MODEL = "IECKAI"
REF_HT = 90  # 参考高度，根据你的截图，这个值是固定的

# 可变参数的范围
# URef 将在每个整数区间内随机取值，例如，当 uref_bin=4 时，实际风速在 4.0 到 5.0 之间
uref_bins = range(4, 19)  # 代表 4-5, 5-6, ..., 18-19 m/s 的风速区间，共15个

# --- 修改点 1: IECturbc 的间隔由5改为1 ---
# 原代码: iec_turbc_values = range(0, 31, 5)
iec_turbc_values = range(0, 31, 1)  # 0, 1, 2, ..., 30，共31个

# --- 修改点 2: Yaw 参数固定为 0 ---
# 原代码: yaw_values = range(0, 31, 5)
# 我们不再需要一个范围，直接在循环外定义一个固定值即可
fixed_yaw = 0

# 输出文件名
output_filename = "load_cases_modified.csv"

# --- 2. 生成工况数据 ---

all_cases = []
case_number = 1

print("正在生成工况...")

# --- 修改点 2 (续): 移除了最内层的 yaw 循环 ---
# 使用两层嵌套循环生成所有参数组合
for uref_bin in uref_bins:
    for iec in iec_turbc_values:
        
        # 为每个工况生成一个唯一的随机种子
        rand_seed = random.randint(10000, 99999)
        
        # 在指定的风速区间内生成一个随机浮点数，并保留两位小数
        # 例如，当 uref_bin = 4 时，uref_random 的范围是 [4.0, 5.0)
        uref_random = random.uniform(uref_bin, uref_bin + 1)
        
        # 创建当前工况行
        # 顺序与你的截图一致
        case_row = [
            case_number,
            rand_seed,
            ANALYSIS_TIME,
            USABLE_TIME,
            TURB_MODEL,
            iec,
            REF_HT,
            f"{uref_random:.2f}",  # 格式化为字符串，保留两位小数
            fixed_yaw  # --- 修改点 2 (续): 此处直接使用固定值 fixed_yaw ---
        ]
        
        # 将当前行添加到总列表中
        all_cases.append(case_row)
        
        # 工况号加一
        case_number += 1

print(f"工况生成完毕，共 {len(all_cases)} 个工况。")

# --- 3. 将数据写入CSV文件 ---

try:
    with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # 写入表头
        header = ['number', 'RandSeed', 'AnalysisTime', 'UsableTime', 'TurbModel', 'IECturbc', 'RefHt', 'URef', 'Yaw']
        writer.writerow(header)
        
        # 写入所有工况数据
        writer.writerows(all_cases)
        
    print(f"成功！已将所有工况保存到文件 '{output_filename}' 中。")

except IOError:
    print(f"错误：无法写入文件 '{output_filename}'。请检查文件是否被其他程序占用或路径是否有写入权限。")