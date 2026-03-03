import subprocess
import os


def run_fst(fst_file_road):
    print("=====" + "run fst" + "=====")
  
    exe_path = "fst_and_results_file/openfast_x64.exe"  
    file_to_process = fst_file_road          

    try:

        result = subprocess.run(
            [exe_path, file_to_process],
            check=True,        
            capture_output=True, 
            text=True           
        )
        print("successful run fst")
        # print('\n')
    except subprocess.CalledProcessError as e:
        print(f"ERROR_1{e.stderr}") 
        print('\n')
    except FileNotFoundError:
        print("ERROR_2") 
        print('\n')
    except Exception as e:
        print(f"NOKOWN{str(e)}") 

