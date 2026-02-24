import subprocess
import os
import shutil

def read_fst(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_fst(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # 提取目录路径
    dir_path = os.path.dirname(file_path)
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # 写入文件
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties_fst(RefHt, URef,IECturbc,Yaw,inflowwind_file_road,ElastoDyn_file_road,RandSeed1,Shear):
    print("=====" + "create fst" + "=====")
    # 读取主参数文件
    input_path_create_fst = "fst_and_results_file/original_file.fst"
    # output_path = "fst_and_results_file/"+ "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  +'TI' + str(IECturbc) + "_Yaw" + str(Yaw) + ".fst"
   
    output_path =  ('fst_and_results_file/'+  "wind_" + str(RandSeed1) + "_" + str(RefHt) + "m_" + str(URef) + "mps_" 
        + 'TI' + str(IECturbc) + "_Yaw" + str(Yaw) + "_Shear" + str(Shear) + ".fst")
    lines = read_fst(input_path_create_fst)

    inflowwind_file_road = str('"../'+ str(inflowwind_file_road) +'"') # inflowwind文件路径
    # hyd_file_road = str('"../'+ str(hyd_file_road) +'"') # hyd文件路径

    # 修改对应行的参数
    ## inflowwind_file_road
    target_line_inflowwind_file_road = 37  # 第38行
    # print(lines[36])
    original_line_inflowwind_file_road = lines[target_line_inflowwind_file_road]
    position_inflowwind_file_road = original_line_inflowwind_file_road.find("InflowFile")
    new_line_inflowwind_file_road =original_line_inflowwind_file_road.replace(original_line_inflowwind_file_road[0:position_inflowwind_file_road], inflowwind_file_road + "    ") # 赋值
    lines[target_line_inflowwind_file_road] = new_line_inflowwind_file_road # 替换
    # print(lines[36])
    ## ElastoDyn_road
    target_line_ElastoDyn_road = 33  # 34
    original_line_inflowwind_file_road = lines[target_line_ElastoDyn_road]
    position_inflowwind_file_road = original_line_inflowwind_file_road.find("EDFile")
    new_line_inflowwind_file_road =original_line_inflowwind_file_road.replace(original_line_inflowwind_file_road[0:position_inflowwind_file_road], ElastoDyn_file_road + "    ") # 赋值
    lines[target_line_ElastoDyn_road] = new_line_inflowwind_file_road # 替换






    # 写回文件
    write_fst(output_path, lines)
    print(f"successful create    {output_path}")
    
    # for ext in ['.u', '.v', '.w']:
    #     src_file = "Wind/"+ str(RefHt) + "m_" + str(URef) + "mps" + "/wind" + ext
    #     dst_file = "Wind_data/" +  "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  + str(IECturbc) + "_" + str(VFlowAng)+ ext
    #     shutil.move(src_file, dst_file)
    # print("=====" + "wind hade moved" + "=====")
    return(output_path)

if __name__ == "__main__":
    modify_properties_fst(90, 2,2,10,'inflowwind_file_road','ElastoDyn_file_road')