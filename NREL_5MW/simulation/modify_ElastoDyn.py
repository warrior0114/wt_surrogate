import os

def read_ElastoDyn(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_ElastoDyn(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # 提取目录路径
    dir_path = os.path.dirname(file_path)
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # 写入文件
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties_ElastoDyn (RefHt, URef, Yaw):
    print("=====" + "create inflowwind" + "=====")
    # 读取风生成参数文件
    input_path_create_inflowwindfile = "fst_and_results_file/NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
    output_path_ElastoDyn = "fst_and_results_file/"+ "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  + "_" +  str(Yaw) + ".dat"
    lines = read_ElastoDyn(input_path_create_inflowwindfile)

   

    # 修改对应行的参数
    ## bts文件路径
    target_line_URef = 33  # 第34行
    original_line_URef = lines[target_line_URef]
    position_URef = original_line_URef.find("NacYaw")
    new_line_URef =original_line_URef.replace(original_line_URef[10:position_URef], str(Yaw) + "   ") # 赋值
    lines[target_line_URef] = new_line_URef # 替换
    # print(lines[21])
 
    # print(lines[6])

    # 写回文件
    write_ElastoDyn(output_path_ElastoDyn, lines)
    print(f"successful create   {output_path_ElastoDyn}")

    output_path_ElastoDyn= "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  + "_" + str(Yaw) +".dat"
    return(output_path_ElastoDyn)

if __name__ == "__main__":
    file=modify_properties_ElastoDyn(120, 12, "30")
    print(file)