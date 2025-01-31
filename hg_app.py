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

# 명령줄 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

# 실행 환경 설정
print(f"Running on {'local' if args.local else 'huggingface'}")
if not args.local:
    print("Skipping GPU setup and Hugging Face-specific settings.")
    
    IP = "0.0.0.0"
    PORT = 7860

else:
    IP = "0.0.0.0"
    PORT = 8080

# GPU 제거 및 CPU 강제 사용
device = torch.device("cpu")
print(f"✅ 실행 장치: {device}")

# 모델 및 작업 함수
def load_model():
    model = torch.load("model.pth", map_location=device)  # CPU에서 실행 가능하도록 변환
    model.to(device)
    return model

# 세션 시작 및 종료 함수
def start_session(req: gr.Request):
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash))
    os.makedirs(save_folder, exist_ok=True)

def end_session(req: gr.Request):
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash))
    shutil.rmtree(save_folder)

# 예제 이미지 및 텍스트 로딩
def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt'):
        txt_list.append(line.strip())
    return txt_list

# 3D 모델 처리 함수
def export_mesh(mesh, save_folder, textured=False):
    filename = 'textured_mesh.glb' if textured else 'white_mesh.glb'
    path = os.path.join(save_folder, filename)
    mesh.export(path, include_normals=textured)
    return path

# HTML 생성
def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    filename = 'textured_mesh.glb' if textured else 'white_mesh.glb'
    related_path = f"./{filename}"
    template_name = './assets/modelviewer-textured-template.html' if textured else './assets/modelviewer-template.html'
    output_html_path = os.path.join(save_folder, f'{uuid.uuid4()}_{filename.replace(".glb", ".html")}')

    with open(template_name, 'r') as f:
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

    return f"""
        <div style='height: {height}; width: 100%;'>
        <iframe src="/static/{output_html_path}" height="{height}" width="100%" frameborder="0"></iframe>
        </div>
    """

# 3D 모델 생성 (GPU 기능 제거)
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
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash)) 
    os.makedirs(save_folder, exist_ok=True)

    generator = torch.Generator().manual_seed(int(seed))
    
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    return path, path_textured, model_viewer_html, model_viewer_html_textured

# Gradio 앱 생성
def build_app():
    with gr.Blocks(title='Hunyuan-3D-2.0') as demo:
        gr.HTML("<h2>Hunyuan3D-2: Scaling Diffusion Models</h2>")
        gr.Textbox(label='Text Prompt', placeholder='Describe the 3D model...')

        btn = gr.Button("Generate Shape Only")
        btn_all = gr.Button("Generate Shape and Texture")

        btn_all.click(
            generation_all,
            inputs=["caption", "image", "steps", "guidance_scale", "seed", "octree_resolution", "check_box_rembg"],
            outputs=["file_out", "file_out2", "html_output1", "html_output2"]
        )

    return demo

# FastAPI 설정 및 실행
if __name__ == '__main__':
    SAVE_DIR = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(SAVE_DIR, exist_ok=True)

    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    # 모델 로드 (GPU 기능 제거됨)
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        HAS_TEXTUREGEN = True
    except:
        HAS_TEXTUREGEN = False

    # FastAPI 서버 실행
    app = FastAPI()
    static_dir = Path('./gradio_cache')
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    demo = build_app()
    demo.queue(max_size=10)
    app = gr.mount_gradio_app(app, demo, path="/")

    uvicorn.run(app, host=IP, port=PORT)
