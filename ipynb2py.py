#!/usr/bin/env python3
import sys
import subprocess
import os

def ipynb_to_py(ipynb_file):
    # ipynb 파일 경로에서 확장자 제거 후 .py로 치환
    base, ext = os.path.splitext(ipynb_file)
    if ext.lower() == '.ipynb':
        # jupyter nbconvert 명령어를 통한 변환
        subprocess.run(['jupyter', 'nbconvert', '--to', 'python', ipynb_file], check=True)
        py_file = base + '.py'
        print(f"{ipynb_file} -> {py_file}")
    else:
        print(f"Skip: {ipynb_file} is not an ipynb file.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ipynb2py.py [list_of_ipynb_files]")
        sys.exit(1)

    for f in sys.argv[1:]:
        ipynb_to_py(f)

