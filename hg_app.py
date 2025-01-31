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
    IP = "0.0.0.0"
    PORT = 7860
else:
    IP = "0.0.0.0"
    PORT = 8080
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

    # rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    # iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    # print(f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# **3D 모델 변환 로직 (CPU 기반)**
def generation_all(
    caption: str,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
    octree_resolution: int,
    req: gr.Request,
):
    print("🚀 3D 모델 생성 시작")
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(Hunyuan3D-2)
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
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(Hunyuan3D-2)
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

uvicorn.run(app, host=IP, port=PORT)
