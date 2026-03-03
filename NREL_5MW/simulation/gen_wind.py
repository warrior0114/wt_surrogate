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
    # Extract directory path
    dir_path = os.path.dirname(file_path)
    # If directory does not exist, create it recursively
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # Write to file
    with open(file_path, "wb") as f:
        f.write(new_content)


def modify_properties(RandSeed1, AnalysisTime, UsableTime, TurbModel, IECturbc, RefHt, URef, Yaw, Shear):
    print("=====" + "create inp and bts" + "=====")
    # Parameter settings
    RandSeed1 = str(RandSeed1) # Random seed
    AnalysisTime = str(AnalysisTime) # Analysis time
    UsableTime = str(UsableTime) # Usable time
    TurbModel = str('"'+ str(TurbModel) +'"') # Turbulence model
    IECturbc1 = IECturbc # Store original value for filename
    IECturbc = str('"'+ str(IECturbc) +'"') # Turbulence Class / Intensity 

    RefHt = str(RefHt) # Hub height
    URef = str(URef) # Reference wind speed
    Yaw = str(Yaw) # Hub yaw angle
    Shear = str(Shear)

    # Read wind generation parameter file
    input_path_create_turbsim_wind = "Wind/original_file.inp"
    output_path = "Wind/"+ str(RefHt) + "m_" + str(URef) + "mps" + "/wind.inp"
    lines = read_inp(input_path_create_turbsim_wind)


    # Modify parameters in corresponding lines
    ## RandSeed1
    target_line_RandSeed1 = 4  # Line 5
    original_line_RandSeed1 = lines[target_line_RandSeed1]
    position_RandSeed1 = original_line_RandSeed1.find("RandSeed1")
    new_line_RandSeed1 = original_line_RandSeed1.replace(original_line_RandSeed1[6:position_RandSeed1], RandSeed1 + "   ") # Assignment
    lines[target_line_RandSeed1] = new_line_RandSeed1 # Replace
    # print(lines[4])

    ## AnalysisTime
    target_line_AnalysisTime = 21  # Line 22
    original_line_AnalysisTime = lines[target_line_AnalysisTime]
    position_AnalysisTime = original_line_AnalysisTime.find("AnalysisTime")
    new_line_AnalysisTime = original_line_AnalysisTime.replace(original_line_AnalysisTime[8:position_AnalysisTime], AnalysisTime + "   ") # Assignment
    lines[target_line_AnalysisTime] = new_line_AnalysisTime # Replace
    # print(lines[21])

    ## UsableTime
    target_line_UsableTime = 22  # Line 23
    original_line_UsableTime = lines[target_line_UsableTime]
    position_UsableTime = original_line_UsableTime.find("UsableTime")
    new_line_UsableTime = original_line_UsableTime.replace(original_line_UsableTime[9:position_UsableTime], UsableTime + "   ") # Assignment
    lines[target_line_UsableTime] = new_line_UsableTime # Replace
    # print(lines[22])

    ## TurbModel (Use default)
    target_line_TurbModel = 30  # Line 31
    original_line_TurbModel = lines[target_line_TurbModel]
    position_TurbModel = original_line_TurbModel.find("TurbModel")
    new_line_TurbModel = original_line_TurbModel.replace(original_line_TurbModel[0:position_TurbModel], TurbModel + "      ") # Assignment
    lines[target_line_TurbModel] = new_line_TurbModel # Replace
    # print(lines[30])

    ## IECturbc 
    target_line_IECturbc = 33  # Line 34
    original_line_IECturbc = lines[target_line_IECturbc]
    position_IECturbc = original_line_IECturbc.find("IECturbc")
    new_line_IECturbc = original_line_IECturbc.replace(original_line_IECturbc[0:position_IECturbc], IECturbc + "           ") # Assignment
    lines[target_line_IECturbc] = new_line_IECturbc # Replace
    # print(lines[33])

   ## Shear (PLExp)
    target_line_Shear = 41  # Line 42
    original_line_Shear = lines[target_line_Shear]
    position_Shear = original_line_Shear.find("PLExp")
    new_line_Shear = original_line_Shear.replace(original_line_Shear[0:position_Shear], Shear + "     ") # Assignment
    lines[target_line_Shear] = new_line_Shear # Replace

    ## RefHt
    target_line_RefHt = 38  # Line 39
    original_line_RefHt = lines[target_line_RefHt]
    position_RefHt = original_line_RefHt.find("RefHt")
    new_line_RefHt = original_line_RefHt.replace(original_line_RefHt[9:position_RefHt], RefHt + "   ") # Assignment
    lines[target_line_RefHt] = new_line_RefHt # Replace
    # print(lines[38])

    ## URef
    target_line_URef = 39  # Line 40
    original_line_URef = lines[target_line_URef]
    position_URef = original_line_URef.find("URef")
    new_line_URef = original_line_URef.replace(original_line_URef[9:position_URef], URef + "   ") # Assignment
    lines[target_line_URef] = new_line_URef # Replace
    # print(lines[39])

    # Write back to file
    write_inp(output_path, lines)
    print(f"Successfully created {output_path}")

    # Run file to generate .bts file
    # Define exe path and path of the file to be processed
    exe_path = "Wind/TurbSim_x64.exe"  # Replace with actual exe path
    file_to_process = output_path          # Replace with actual file path

    try:
        # Run command using subprocess
        result = subprocess.run(
            [exe_path, file_to_process],
            check=True,        # Check return code, raise exception if non-zero
            capture_output=True, # Optional: Capture output
            text=True           # Optional: Return output as text
        )
        print("Successfully created bts")
        input_bts_file =  "Wind/"+ str(RefHt) + "m_" + str(URef) + "mps" + "/wind.bts"
        print(input_bts_file)
        output_bts_file = ('Wind_bts/'+  "wind_" + str(RandSeed1) + "_" + str(RefHt) + "m_" + str(URef) + "mps_" 
        + 'TI' + str(IECturbc1) + "_Yaw" + str(Yaw) + "_Shear" + str(Shear) + ".bts")
                        
        shutil.move(input_bts_file, output_bts_file)
        
        
        # print('\n')
    except subprocess.CalledProcessError as e:
        print(f"ERROR_1_***{e.stderr}") # Error: Program returned non-zero status code. Details
        print('\n')
    except FileNotFoundError:
        print("ERROR_2") # Error: Specified exe file not found. Please check if path is correct.
        print('\n')
    except Exception as e:
        print(f"UNKNOWN {str(e)}") # Unknown error
        
    # Return bts file path
    return (RefHt, URef, output_bts_file)
    

if __name__ == "__main__":
    # Generate bts file
    RefHt = 90
    URef = 10
    RandSeed1 = 1
    AnalysisTime = 3600
    UsableTime = 3600
    TurbModel = "IECKAI"
    IECturbc1 = 0.1

    modify_properties(10,20,10,"IECKAI","A",90,7,10,0.1)