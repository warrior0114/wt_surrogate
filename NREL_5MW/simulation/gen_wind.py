import subprocess
import os
from modify_map import modify_wind_data
import shutil


def read_inp(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_inp(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # 提取目录路径
    dir_path = os.path.dirname(file_path)
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # 写入文件
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties(RandSeed1, AnalysisTime, UsableTime, TurbModel, IECturbc, RefHt, URef,Yaw,Shear):
    print("=====" + "create inp and bts" + "=====")
    # 参数设置
    RandSeed1 = str(RandSeed1) # 随机种子
    AnalysisTime = str(AnalysisTime) # 分析时间
    UsableTime = str(UsableTime) # 可用时间
    TurbModel = str('"'+ str(TurbModel) +'"') # 湍流模型
    IECturbc1 = IECturbc #
    IECturbc = str('"'+ str(IECturbc) +'"') # 湍流等级 

    RefHt = str(RefHt) # 轮毂高度
    URef = str(URef) # 湍流风速
    Yaw = str(Yaw) # 轮毂偏航角
    Shear =str(Shear)

    # 读取风生成参数文件
    input_path_create_turbsim_wind = "Wind/original_file.inp"
    output_path = "Wind/"+ str(RefHt) + "m_" + str(URef) + "mps" + "/wind.inp"
    lines = read_inp(input_path_create_turbsim_wind)


    # 修改对应行的参数
    ## RandSeed1
    target_line_RandSeed1 = 4  # 第5行
    original_line_RandSeed1 = lines[target_line_RandSeed1]
    position_RandSeed1 = original_line_RandSeed1.find("RandSeed1")
    new_line_RandSeed1 =original_line_RandSeed1.replace(original_line_RandSeed1[6:position_RandSeed1], RandSeed1 + "   ") # 赋值
    lines[target_line_RandSeed1] = new_line_RandSeed1 # 替换
    # print(lines[4])

    ## AnalysisTime
    target_line_AnalysisTime = 21  # 第22行
    original_line_AnalysisTime = lines[target_line_AnalysisTime]
    position_AnalysisTime = original_line_AnalysisTime.find("AnalysisTime")
    new_line_AnalysisTime =original_line_AnalysisTime.replace(original_line_AnalysisTime[8:position_AnalysisTime], AnalysisTime + "   ") # 赋值
    lines[target_line_AnalysisTime] = new_line_AnalysisTime # 替换
    # print(lines[21])

    ## UsableTime
    target_line_UsableTime = 22  # 第23行
    original_line_UsableTime = lines[target_line_UsableTime]
    position_UsableTime = original_line_UsableTime.find("UsableTime")
    new_line_UsableTime =original_line_UsableTime.replace(original_line_UsableTime[9:position_UsableTime], UsableTime + "   ") # 赋值
    lines[target_line_UsableTime] = new_line_UsableTime # 替换
    # print(lines[22])

    ## TurbModel 用默认的就行
    target_line_TurbModel = 30  # 第31行
    original_line_TurbModel = lines[target_line_TurbModel]
    position_TurbModel = original_line_TurbModel.find("TurbModel")
    new_line_TurbModel =original_line_TurbModel.replace(original_line_TurbModel[0:position_TurbModel], TurbModel + "      ") # 赋值
    lines[target_line_TurbModel] = new_line_TurbModel # 替换
    # print(lines[30])

    ## IECturbc 
    target_line_IECturbc = 33  # 第34行
    original_line_IECturbc = lines[target_line_IECturbc]
    position_IECturbc = original_line_IECturbc.find("IECturbc")
    new_line_IECturbc =original_line_IECturbc.replace(original_line_IECturbc[0:position_IECturbc], IECturbc + "           ") # 赋值
    lines[target_line_IECturbc] = new_line_IECturbc # 替换
    # print(lines[33])

   ## sheer 
    target_line_IECturbc = 41  # 第42行
    original_line_IECturbc = lines[target_line_IECturbc]
    position_IECturbc = original_line_IECturbc.find("PLExp")
    new_line_IECturbc =original_line_IECturbc.replace(original_line_IECturbc[0:position_IECturbc], Shear + "     ") # 赋值
    lines[target_line_IECturbc] = new_line_IECturbc # 替换

    ## RefHt
    target_line_RefHt = 38  # 第39行
    original_line_RefHt = lines[target_line_RefHt]
    position_RefHt = original_line_RefHt.find("RefHt")
    new_line_RefHt =original_line_RefHt.replace(original_line_RefHt[9:position_RefHt], RefHt + "   ") # 赋值
    lines[target_line_RefHt] = new_line_RefHt # 替换
    # print(lines[38])

    ## URef
    target_line_URef = 39  # 第40行
    original_line_URef = lines[target_line_URef]
    position_URef = original_line_URef.find("URef")
    new_line_URef =original_line_URef.replace(original_line_URef[9:position_URef], URef + "   ") # 赋值
    lines[target_line_URef] = new_line_URef # 替换
    # print(lines[39])

    # 写回文件
    write_inp(output_path, lines)
    print(f"successful create   {output_path}")

    # 运行文件，生成bts文件
    # 定义exe路径和待处理文件路径
    exe_path = "Wind/TurbSim_x64.exe"  # 替换为实际的exe路径
    file_to_process = output_path          # 替换为实际的文件路径

    try:
        # 使用subprocess运行命令
        result = subprocess.run(
            [exe_path, file_to_process],
            check=True,        # 检查返回码，非零时抛出异常
            capture_output=True, # 可选：捕获输出
            text=True           # 可选：以文本形式返回输出
        )
        print("successful create bts")
        input_bts_file =  "Wind/"+ str(RefHt) + "m_" + str(URef) + "mps" + "/wind.bts"
        print(input_bts_file)
        output_bts_file = ('Wind_bts/'+  "wind_" + str(RandSeed1) + "_" + str(RefHt) + "m_" + str(URef) + "mps_" 
        + 'TI' + str(IECturbc1) + "_Yaw" + str(Yaw) + "_Shear" + str(Shear) + ".bts")
                        
        shutil.move(input_bts_file, output_bts_file)
        
        
        # print('\n')
    except subprocess.CalledProcessError as e:
        print(f"ERROR_1_***{e.stderr}") # 错误：程序返回非零状态码。详细信息
        print('\n')
    except FileNotFoundError:
        print("ERROR_2") # 错误：找不到指定的exe文件。请检查路径是否正确。
        print('\n')
    except Exception as e:
        print(f"NOKOWN{str(e)}") # 未知错误 .
        
    # 返回bts文件路径
    return (RefHt, URef, output_bts_file)
    

if __name__ == "__main__":
    # 生成bts文件
    RefHt = 90
    URef = 10
    RandSeed1 = 1
    AnalysisTime = 3600
    UsableTime = 3600
    TurbModel = "IECKAI"
    IECturbc1 = 0.1

    modify_properties(10,20,10,"IECKAI","A",90,7,10,0.1)
    