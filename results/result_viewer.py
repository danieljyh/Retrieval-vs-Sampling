import os

def recursive_search(path):
    f_list = os.listdir(path)
    for f in f_list:
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            recursive_search(full_path)
        elif os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    first_line = file.readline().strip()
                    print(f"{full_path}: {first_line}")
            except Exception as e:
                print(f"Failed to read {full_path}: {e}")

recursive_search(os.getcwd())
