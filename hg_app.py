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


def _gen_shape(
    caption: str,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
    octree_resolution: int,
    check_box_rembg: bool,
    req: gr.Request,
):
    if caption: print('prompt is', caption)
    save_folder = os.path.join(SAVE_DIR, str(req.session_hash)) 
    os.makedirs(save_folder, exist_ok=True)

    stats = {}
    time_meta = {}
    start_time_0 = time.time()

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'input.png'))

    print(f"[{datetime.now()}][HunYuan3D-2]]", str(req.session_hash), image.mode)
    if check_box_rembg or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['rembg'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    time_meta['image_to_textured_3d'] = {'total': time.time() - start_time}
    time_meta['total'] = time.time() - start_time_0
    stats['time'] = time_meta
    
    torch.cuda.empty_cache()
    return mesh, save_folder, image

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
    mesh, save_folder, image = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        req=req
    )
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    torch.cuda.empty_cache()
    return (
        path,
        path_textured, 
        model_viewer_html,
        model_viewer_html_textured,
    )

def shape_generation(
    caption: str,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
    octree_resolution: int,
    check_box_rembg: bool,
    req: gr.Request,
):
    mesh, save_folder, image = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        req=req,
    )

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    return (
        path,
        model_viewer_html,
    )


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
      <a href="https://github.com/Tencent/Hunyuan3D-2?tab=readme-ov-file#blender-addon"> Blender Addon</a> &ensp;
    </div>
    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', delete_cache=(1000,1000)) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: 3D model of a fantasy world for metaverse game')
                    
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=50, value=25, label="Steps", interactive=True)
                        guidance_scale = gr.Slider(minimum=1, maximum=20, value=7, label="Guidance scale", interactive=True)
                        seed = gr.Slider(minimum=0, maximum=9999, value=42, label="Seed", interactive=True)
                        octree_resolution = gr.Slider(minimum=4, maximum=10, value=7, label="Octree Resolution", interactive=True)

                    with gr.Row():
                        generate_btn = gr.Button(value='Generate')
                
                gr.Examples(examples=get_example_img_list(), fn=generation_all, inputs=[caption, image, steps, guidance_scale, seed, octree_resolution, check_box_rembg], outputs=["file", "file", "html", "html"])

    return demo

app = build_app()

if __name__ == "__main__":
    app.launch(debug=True)
