import argparse
import os
import subprocess
import shlex
import torch
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from glob import glob
from PIL import Image

# **CLI 인자 설정**
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

# **GPU 대신 CPU 사용**
device = torch.device("cpu")
print(f"🔥 실행 장치: {device}")

if not args.local:
    print("🔹 Hugging Face Spaces에서 실행 중")
    
    # **custom_rasterizer 설치**
    print("🔹 custom_rasterizer 설치 중...")
    subprocess.run(shlex.split("pip install --no-cache-dir custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"), check=True)

    IP = "0.0.0.0"
    PORT = 7860
else:
    IP = "0.0.0.0"
    PORT = 8080

# **예제 데이터 불러오기**
def get_example_img_list():
    return sorted(glob('./assets/example_images/*.png'))

def get_example_txt_list():
    with open('./assets/example_prompts.txt') as f:
        return [line.strip() for line in f]

# **3D 모델 변환 로직 (CPU 기반)**
def generation_all(
    caption: str,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
    octree_resolution: int,
    check_box_rembg: bool,
    req: gr.Request,
):
    print("🚀 3D 모델 생성 시작")
    
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash)) 
    os.makedirs(save_folder, exist_ok=True)

    generator = torch.Generator(device=device)  # CPU 기반 생성기
    generator = generator.manual_seed(int(seed))

    # **3D 생성 모델 적용**
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    print("🎨 텍스처 생성 중...")
    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    print("✅ 3D 모델 생성 완료")
    return (path, path_textured, model_viewer_html, model_viewer_html_textured)

# **FastAPI 및 Gradio 설정**
app = FastAPI()
static_dir = Path('./gradio_cache')
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

demo = gr.Blocks(title='Hunyuan3D-2.0')
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(CURRENT_DIR, args.cache_path)
    os.makedirs(SAVE_DIR, exist_ok=True)

    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    # **모델 불러오기 (CPU)**
    # 모델 로드 부분 수정
from hy3dgen.text2image import HunyuanDiTPipeline

# CPU 강제 적용
t2i_worker = HunyuanDiTPipeline.from_pretrained(
    'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
    device_map="cpu"  # ✅ GPU 없이 실행 가능하도록 설정
).to(device)

    uvicorn.run(app, host=IP, port=PORT)
