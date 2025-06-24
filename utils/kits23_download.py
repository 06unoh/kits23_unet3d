import os
import subprocess
import shutil

def get_dataset():   
    default_path = os.path.expanduser("~/.kits23/cases/")
    target_path = "./datasets/kits23/"

    # 데이터 없으면 다운로드 실행
    if not os.path.exists(target_path):
        subprocess.run(["kits23_download_data"], check=True)
        os.makedirs(target_path, exist_ok=True)
        shutil.move(default_path, target_path)
        print("Download Done")
