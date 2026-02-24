import os

def read_inflowwind(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_inflowwind(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # 提取目录路径
    dir_path = os.path.dirname(file_path)
    # 如果目录不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # 写入文件
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties_inflowwind (RefHt, URef, btsfile_load):
    print("=====" + "create inflowwind" + "=====")
    # 读取风生成参数文件
    input_path_create_inflowwindfile = "Wind/NRELOffshrBsline5MW_InflowWind_12mps.dat"
    output_path_inflowwindfile = "inflowwind_file/"+ str(RefHt) + "m_" + str(URef) + "mps" + ".dat"
    lines = read_inflowwind(input_path_create_inflowwindfile)

    btsfile_load = str('"../'+ str(btsfile_load) +'"') # bts文件路径

    # 修改对应行的参数
    ## bts文件路径
    target_line_btsfile_load = 21  # 第22行
    original_line_btsfile_load = lines[target_line_btsfile_load]
    position_btsfile_load = original_line_btsfile_load.find("FileName_BTS")
    new_line_btsfile_load =original_line_btsfile_load.replace(original_line_btsfile_load[0:position_btsfile_load], btsfile_load + "    ") # 赋值
    lines[target_line_btsfile_load] = new_line_btsfile_load # 替换
    # print(lines[21])
 
    # print(lines[6])

    # 写回文件
    write_inflowwind(output_path_inflowwindfile, lines)
    print(f"successful create   {output_path_inflowwindfile}")

    return(output_path_inflowwindfile)

if __name__ == "__main__":
    modify_properties_inflowwind(120, 12, "BTS_120m.bts")