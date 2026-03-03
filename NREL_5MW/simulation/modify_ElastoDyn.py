import os

def read_ElastoDyn(file_path):
    with open(file_path, "rb") as f:
        raw_content = f.read()
        text_content = raw_content.decode("utf-8")
    return text_content.splitlines(keepends=True)

def write_ElastoDyn(file_path, lines):
    new_content = "".join(lines).encode("utf-8")
    # Extract directory path
    dir_path = os.path.dirname(file_path)
    # If directory does not exist, create it recursively
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # Write to file
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties_ElastoDyn(RefHt, URef, Yaw):
    print("=====" + "create elastodyn" + "=====")
    # Read ElastoDyn parameter file
    input_path_create_inflowwindfile = "fst_and_results_file/NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
    output_path_ElastoDyn = "fst_and_results_file/"+ "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  + "_" +  str(Yaw) + ".dat"
    lines = read_ElastoDyn(input_path_create_inflowwindfile)

   

    # Modify parameters in corresponding lines
    ## NacYaw (Nacelle Yaw)
    target_line_URef = 33  # Line 34
    original_line_URef = lines[target_line_URef]
    position_URef = original_line_URef.find("NacYaw")
    new_line_URef = original_line_URef.replace(original_line_URef[10:position_URef], str(Yaw) + "   ") # Assignment
    lines[target_line_URef] = new_line_URef # Replace
    # print(lines[21])
 
    # print(lines[6])

    # Write back to file
    write_ElastoDyn(output_path_ElastoDyn, lines)
    print(f"Successfully created   {output_path_ElastoDyn}")

    output_path_ElastoDyn= "wind_" + str(RefHt) + "m_" + str(URef) + "mps_"  + "_" + str(Yaw) +".dat"
    return(output_path_ElastoDyn)

if __name__ == "__main__":
    file=modify_properties_ElastoDyn(120, 12, "30")
    print(file)