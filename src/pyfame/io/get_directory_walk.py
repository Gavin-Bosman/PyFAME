import os

def get_directory_walk(root_path:str, with_sub_dirs:bool = False) -> list[str]:
    files_to_process = []
    single_file = False

    if os.path.isfile(root_path):
        single_file = True

    if single_file:
        files_to_process.append(root_path)
    elif not with_sub_dirs:
        files_to_process = [root_path + "\\" + file for file in os.listdir(root_path)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(root_path, topdown=True) 
                            for file in files]
    
    return files_to_process