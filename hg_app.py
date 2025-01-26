import argparse
import os
import shutil
import time
from glob import glob
from pathlib import Path
from PIL import Image
from datetime import datetime
import uuid
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

if not args.local:
    import subprocess
    import shlex
    import os

    

    def get_system_info():
    # 시스템 정보 확인
        system_info = {
        'os': platform.system(),  # 운영 체제 (예: 'Linux', 'Windows')
        'architecture': platform.architecture()[0],  # 시스템 아키텍처 (예: '64bit', '32bit')
        'python_version': platform.python_version(),  # Python 버전 (예: '3.8.10')
        'machine': platform.machine(),  # 머신 아키텍처 (예: 'x86_64', 'arm64')
        'processor': platform.processor()  # 프로세서 정보
        }
    return system_info

    def find_compatible_whl(system_info):
    # Python 버전과 아키텍처에 맞는 .whl 파일을 찾기
        python_version = system_info['python_version']
        architecture = system_info['architecture']
    
    # 예시로, `custom_rasterizer`의 경우 `cp310` (Python 3.10) 과 `x86_64` 아키텍처에 맞는 `.whl` 파일을 찾음
        whl_filename = f"custom_rasterizer-0.1-cp{python_version.replace('.', '')}-cp{python_version.replace('.', '')}-linux_{architecture}.whl"

    # 파일 경로 확인 (현재 경로에서 찾기)
    if os.path.exists(whl_filename):
        return whl_filename
    else:
        print(f"Warning: {whl_filename} does not exist in the current directory.")
        return None
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install {whl_filename}"), check=True)   

def install_whl(whl_file):
    if whl_file:
        try:
            print(f"Installing {whl_file}...")
            subprocess.run(["pip", "install", whl_file], check=True)
            print("Installation successful!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {whl_file}: {e}")
    else:
        print("No compatible .whl file found to install.")

if __name__ == "__main__":
    system_info = get_system_info()
    print(f"Detected system info: {system_info}")
    
    whl_file = find_compatible_whl(system_info)
    install_whl(whl_file)

    IP = "0.0.0.0"
    PORT = 7860
else:
    IP = "0.0.0.0"
    PORT = 8080

# Initialize directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cache_path)
os.makedirs(SAVE_DIR, exist_ok=True)

def start_session(req: gr.Request):
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash))
    os.makedirs(save_folder, exist_ok=True)

def end_session(req: gr.Request):
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash))
    shutil.rmtree(save_folder)

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt'):
        txt_list.append(line.strip())
    return txt_list

def export_mesh(mesh, save_folder, textured=False):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.glb')
    else:
        path = os.path.join(save_folder, f'white_mesh.glb')
    mesh.export(path, include_normals=textured)
    return path

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'{uuid.uuid4()}_textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'{uuid.uuid4()}_white_mesh.html')

    with open(os.path.join(CURRENT_DIR, template_name), 'r') as f:
        template_html = f.read()
        obj_html = f"""
            <div class="column is-mobile is-centered">
                <model-viewer style="height: {height - 10}px; width: {width}px;" rotation-per-second="10deg" id="modelViewer"
                    src="{related_path}/" disable-tap 
                    environment-image="neutral" auto-rotate camera-target="0m 0m 0m" orientation="0deg 0deg 170deg" shadow-intensity=".9"
                    ar auto-rotate camera-controls>
                </model-viewer>
            </div>
            """

    with open(output_html_path, 'w') as f:
        f.write(template_html.replace('<model-viewer>', obj_html))

    output_html_path = output_html_path.replace(SAVE_DIR + '/', '')
    iframe_tag = f'<iframe src="/static/{output_html_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html {output_html_path}, {os.path.exists(output_html_path)}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

# Gradio 앱을 FastAPI와 함께 실행
def build_app():
    title_html = """
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
    Hunyuan3D-2: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    <div align="center">
      <a href="https://github.com/tencent/Hunyuan3D-2">Github Page</a> &ensp; 
      <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
      <a href="https://arxiv.org/abs/2501.12202">Technical Report</a> &ensp;
      <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Models</a> &ensp;
    </div>
    """
    
    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0') as demo:
        gr.HTML(title_html)

        # 사용자 입력 필드
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                text_input = gr.Textbox(label="Enter Description", placeholder="Type a description here...")

        # 버튼 및 그리드 출력
        with gr.Row():
            with gr.Column():
                generate_button = gr.Button("Generate 3D Model")
                output_html = gr.HTML(label="Generated 3D Model")

        # 결과 표시
        generate_button.click(fn=process_model, inputs=[image_input, text_input], outputs=output_html)

        demo.load(start_session)
        demo.unload(end_session)

    return demo

def process_model(image, description):
    # 모델 생성 함수, 이미지를 사용하고 텍스트로 설명을 받음
    print("Processing model with image and description...")
    save_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(save_folder, exist_ok=True)
    
    # 이미지와 텍스트로 모델 생성하는 코드 추가
    # 예를 들어, 모델을 텍스쳐화된 형태로 생성하는 함수 (텍스트에 대한 처리 추가 가능)
    
    # 모델 결과 HTML 생성
    result_html = build_model_viewer_html(save_folder, textured=True)
    return result_html

if __name__ == '__main__':
    # FastAPI app 생성
    app = FastAPI()
    static_dir = Path('./gradio_cache')
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    demo = build_app()
    demo.queue(max_size=10)
    
    # Gradio 앱을 FastAPI와 통합
    app = gr.mount_gradio_app(app, demo, path="/")
    
    uvicorn.run(app, host=IP, port=PORT)
