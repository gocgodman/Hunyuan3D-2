# pip install gradio==3.39.0
import os
import subprocess
def install_cuda_toolkit():
    # CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
    CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run"
    CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
    subprocess.call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
    subprocess.call(["chmod", "+x", CUDA_TOOLKIT_FILE])
    subprocess.call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])

    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
    os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
        os.environ["CUDA_HOME"],
        "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
    )
    # Fix: arch_list[-1] += '+PTX'; IndexError: list index out of range
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"

# install_cuda_toolkit()
os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
os.system("cd /home/user/app/hy3dgen/texgen/custom_rasterizer && pip install .")
os.system("cd /home/user/app/hy3dgen/texgen/custom_rasterizer && CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' python setup.py install")

import shutil
import time
from glob import glob

import gradio as gr
import torch

import spaces

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png'))


def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt'):
        txt_list.append(line.strip())
    return txt_list


def gen_save_folder(max_size=60):
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if not _.startswith("."))
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"remove {SAVE_DIR}/{(cur_id + 1) % max_size} success !!!")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"mkdir {save_folder} suceess !!!")
    return save_folder


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
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')

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

    iframe_tag = f'<iframe src="file/{output_html_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html {output_html_path}, {os.path.exists(output_html_path)}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

@spaces.GPU(duration=60)
def _gen_shape(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
):
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {}
    time_meta = {}
    start_time_0 = time.time()

    image_path = ''
    if image is None:
        start_time = time.time()
        image = t2i_worker(caption)
        time_meta['text2image'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'input.png'))

    print(image.mode)
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
    return mesh, save_folder

@spaces.GPU(duration=80)
def generation_all(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False
):
    mesh, save_folder = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg
    )
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    return (
        gr.update(value=path, visible=True),
        gr.update(value=path_textured, visible=True),
        model_viewer_html,
        model_viewer_html_textured,
    )

@spaces.GPU(duration=30)
def shape_generation(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
):
    mesh, save_folder = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg
    )

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    return (
        gr.update(value=path, visible=True),
        model_viewer_html,
    )


def build_app():
    title_html = """
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 20px">
    
    Hunyuan3D-2: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    <div align="center">
      <a href="https://github.com/tencent/Hunyuan3D-1">Github Page</a> &ensp; 
      <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
      <a href="https://arxiv.org/pdf/2411.02293">Technical Report</a> &ensp;
      <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Models</a> &ensp;
    </div>
    """
    css = """
    .json-output {
        height: 578px;
    }
    .json-output .json-holder {
        height: 538px;
        overflow-y: scroll;
    }
    """

    with gr.Blocks(theme=gr.themes.Base(), css=css, title='Hunyuan-3D-2.0') as demo:
        if not gr.__version__.startswith('4'):
            gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

                    with gr.Tab('Text Prompt', id='tab_txt_prompt') as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')

                with gr.Accordion('Advanced Options', open=False):
                    num_steps = gr.Slider(maximum=50, minimum=20, value=30, step=1, label='Inference Steps')
                    octree_resolution = gr.Dropdown([256, 384, 512], value=256, label='Octree Resolution')
                    cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
                    seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')

                with gr.Group():
                    btn = gr.Button(value='Generate Shape Only', variant='primary')
                    btn_all = gr.Button(value='Generate Shape and Texture', variant='primary')

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh') as mesh1:
                        html_output1 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Generated Textured Mesh') as mesh2:
                        html_output2 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')

            with gr.Column(scale=2):
                with gr.Tabs() as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery') as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label="Image Prompts", examples_per_page=18)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery') as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label="Text Prompts", examples_per_page=18)

        tab_gi.select(fn=lambda: gr.update(selected='tab_img_prompt'), outputs=tabs_prompt)
        tab_gt.select(fn=lambda: gr.update(selected='tab_txt_prompt'), outputs=tabs_prompt)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
            ],
            outputs=[file_out, html_output1]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[file_out],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
            ],
            outputs=[file_out, file_out2, html_output1, html_output2]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[file_out, file_out2],
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--cache-path', type=str, default='./gradio_cache')
    args = parser.parse_args()

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    HTML_OUTPUT_PLACEHOLDER = """
    <div style='height: 596px; width: 100%; border-radius: 8px; border-color: #e5e7eb; order-style: solid; border-width: 1px;'></div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    # t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled')
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    demo = build_app()
    demo.queue().launch()
