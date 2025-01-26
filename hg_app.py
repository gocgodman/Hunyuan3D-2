import argparse
import os
import shutil
import time
import uuid
from glob import glob
from pathlib import Path
from PIL import Image
from datetime import datetime
import torch
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Gradio 앱 실행에 필요한 경로 설정
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

# 기본 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cache_path)
os.makedirs(SAVE_DIR, exist_ok=True)

# 모델 불러오기
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# 각 파이프라인 로드
rmbg_worker = BackgroundRemover()
texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
floater_remove_worker = FloaterRemover()
degenerate_face_remove_worker = DegenerateFaceRemover()
face_reduce_worker = FaceReducer()

# 3D 메쉬 생성 및 파일 내보내기 함수
def export_mesh(mesh, save_folder, textured=False):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.glb')
    else:
        path = os.path.join(save_folder, f'white_mesh.glb')
    mesh.export(path, include_normals=textured)
    return path

# 모델 뷰어 HTML 생성 함수
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

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

# 3D 모델 생성 로직
def _gen_shape(caption: str, image: Image.Image, steps: int, guidance_scale: float, seed: int, octree_resolution: int, check_box_rembg: bool, req: gr.Request):
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash)) 
    os.makedirs(save_folder, exist_ok=True)

    # 텍스트나 이미지로부터 3D 모델 생성
    if image is None:
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`.")
    image.save(os.path.join(save_folder, 'input.png'))

    if check_box_rembg:
        image = rmbg_worker(image.convert('RGB'))
    
    image.save(os.path.join(save_folder, 'rembg.png'))

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    mesh = i23d_worker(image=image, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator, octree_resolution=octree_resolution)[0]
    
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    return mesh, save_folder, image

def generation_all(caption: str, image: Image.Image, steps: int, guidance_scale: float, seed: int, octree_resolution: int, check_box_rembg: bool, req: gr.Request):
    mesh, save_folder, image = _gen_shape(caption, image, steps, guidance_scale, seed, octree_resolution, check_box_rembg, req)
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    return path, path_textured, model_viewer_html, model_viewer_html_textured

# Gradio UI 구성
def build_app():
    with gr.Blocks() as demo:
        with gr.Row():
            image = gr.Image(label='Input Image', type='pil', height=290)
            caption = gr.Textbox(label='Text Prompt', placeholder='Example: A 3D model of a cute cat')

            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

            num_steps = gr.Slider(maximum=50, minimum=20, value=50, step=1, label='Inference Steps')
            octree_resolution = gr.Dropdown([256, 384, 512], value=256, label='Octree Resolution')
            cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
            seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')

            btn_all = gr.Button(value='Generate Shape and Texture', variant='primary')

            file_out = gr.DownloadButton(label="Download White Mesh", interactive=False)
            file_out2 = gr.DownloadButton(label="Download Textured Mesh", interactive=False)

            html_output1 = gr.HTML(label='3D Model')
            html_output2 = gr.HTML(label='Textured Model')

        btn_all.click(
            generation_all,
            inputs=[caption, image, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg],
            outputs=[file_out, file_out2, html_output1, html_output2]
        )

    return demo

if __name__ == '__main__':
    app = FastAPI()
    static_dir = Path('./gradio_cache')
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    demo = build_app()
    demo.queue(max_size=10)
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
