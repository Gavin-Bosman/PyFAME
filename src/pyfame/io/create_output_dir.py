import os

def create_output_dir(root_path:str, dir_name:str) -> str:
    if not os.path.isdir(root_path + f"\\{dir_name}"):
        os.mkdir(root_path + f"\\{dir_name}")
        
    return root_path + f"\\{dir_name}"