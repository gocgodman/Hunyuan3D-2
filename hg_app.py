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

# Argument parsing for setting the port, cache path, and enabling text-to-3D
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

print(f"Running on {'local' if args.local else 'huggingface'}")

# Directory setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, args.cache_path)
os.makedirs(SAVE_DIR, exist_ok=True)

# Helper functions
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


@gr.outputs.GPU(duration=100)
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

    if check_box_rembg or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['rembg'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'rembg.png'))

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

# Gradio app build
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

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0') as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

                    with gr.Tab('Text Prompt', id='tab_txt_prompt') as tab_tp:
                        caption = gr.Textbox(label='Text Prompt', placeholder='Generate 3D model from text')

                with gr.Accordion('Advanced Options', open=False):
                    num_steps = gr.Slider(maximum=50, minimum=20, value=50, step=1, label='Inference Steps')
                    octree_resolution = gr.Dropdown([256, 384, 512], value=256, label='Octree Resolution')
                    cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
                    seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')

                with gr.Group():
                    btn = gr.Button(value='Generate Shape Only', variant='primary')
                    btn_all = gr.Button(value='Generate Shape and Texture', variant='primary')

                with gr.Group():
                    file_out = gr.DownloadButton(label="Download White Mesh", interactive=False)
                    file_out2 = gr.DownloadButton(label="Download Textured Mesh", interactive=False)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh') as mesh1:
                        html_output1 = gr.HTML("<div style='height: 596px; width: 100%;'></div>", label='Output')
                    with gr.Tab('Generated Textured Mesh') as mesh2:
                        html_output2 = gr.HTML("<div style='height: 596px; width: 100%;'></div>", label='Output')

        btn.click(
            shape_generation,
            inputs=[caption, image, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg],
            outputs=[file_out, html_output1]
        )

        btn_all.click(
            generation_all,
            inputs=[caption, image, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg],
            outputs=[file_out, file_out2, html_output1, html_output2]
        )

    return demo


# Running Gradio app
demo = build_app()
demo.queue(max_size=10)
demo.launch(server_port=args.port, server_name="0.0.0.0", share=True)
