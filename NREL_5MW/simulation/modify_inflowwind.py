import os

def read_inflowwind(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_inflowwind(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # Extract directory path
    dir_path = os.path.dirname(file_path)
    # If directory does not exist, create it recursively
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # Write to file
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties_inflowwind (RefHt, URef, btsfile_load):
    print("=====" + "create inflowwind" + "=====")
    # Read wind generation parameter file
    input_path_create_inflowwindfile = "Wind/NRELOffshrBsline5MW_InflowWind_12mps.dat"
    output_path_inflowwindfile = "inflowwind_file/"+ str(RefHt) + "m_" + str(URef) + "mps" + ".dat"
    lines = read_inflowwind(input_path_create_inflowwindfile)

    btsfile_load = str('"../'+ str(btsfile_load) +'"') # bts file path

    # Modify parameters in corresponding lines
    ## bts file path
    target_line_btsfile_load = 21  # Line 22
    original_line_btsfile_load = lines[target_line_btsfile_load]
    position_btsfile_load = original_line_btsfile_load.find("FileName_BTS")
    new_line_btsfile_load = original_line_btsfile_load.replace(original_line_btsfile_load[0:position_btsfile_load], btsfile_load + "    ") # Assignment
    lines[target_line_btsfile_load] = new_line_btsfile_load # Replace
    # print(lines[21])
 
    # print(lines[6])

    # Write back to file
    write_inflowwind(output_path_inflowwindfile, lines)
    print(f"Successfully created   {output_path_inflowwindfile}")

    return(output_path_inflowwindfile)

if __name__ == "__main__":
    modify_properties_inflowwind(120, 12, "BTS_120m.bts")