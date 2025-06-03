import os

def create_output_directory(root_path:str, dir_name:str) -> str:
    if not os.path.isdir(root_path + f"\\{dir_name}"):
        os.mkdir(root_path + f"\\{dir_name}")
        
    return root_path + f"\\{dir_name}"

def get_directory_walk(root_path:str, with_sub_dirs:bool = False) -> list[str] | tuple[list[str],list[str]]:
    files_to_process = []
    sub_dirs = []
    single_file = False

    if os.path.isfile(root_path):
        single_file = True

    if single_file:
        files_to_process.append(root_path)
    elif not with_sub_dirs:
        files_to_process = [root_path + "\\" + file for file in os.listdir(root_path)]
    else: 
        for path, dirs, files in os.walk(root_path, topdown=True):
            for dir in dirs:
                full_path = os.path.join(path, dir)
                rel_path = os.path.relpath(full_path, root_path)
                sub_dirs.append(rel_path)
            for file in files:
                files_to_process.append(os.path.join(path, file))

    if with_sub_dirs:
        return (sub_dirs, files_to_process)
    else:
        return files_to_process
    
def map_directory_structure(input_dir:str, output_dir:str, with_sub_dirs:bool = False) -> None:
    dir_names = None
    if with_sub_dirs:
        (dir_names, files) = get_directory_walk(input_dir, with_sub_dirs)
    else:
        files = get_directory_walk(input_dir, with_sub_dirs)

    if dir_names is not None:
        for dir in dir_names:
            create_output_directory(output_dir, dir)