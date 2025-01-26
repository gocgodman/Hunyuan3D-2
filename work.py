import gradio as gr
import torch
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
import os
# 구글 드라이브에 모델을 저장할 폴더 생성
os.makedirs(drive_model_folder, exist_ok=True)

def image_to_3d(image, num_steps, cfg_scale, seed, octree_resolution, remove_bg):
    model_path = os.path.join(drive_model_folder, 'Hunyuan3D-2')  # 구글 드라이브 내 모델 경로 설정

    # 모델이 로드되지 않은 경우 구글 드라이브에서 다운로드
    if not os.path.exists(model_path):
        print("모델을 다운로드 중...")
        # 모델 다운로드 과정 (구글 드라이브에 모델 파일을 미리 업로드해야 합니다)
        # 모델 파일을 다운로드하는 코드가 필요하면 여기 추가

    # 이미지 전처리
    
    # 3D 생성 파이프라인 설정
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    mesh = pipeline(image=image, num_inference_steps=num_steps, mc_algo='mc',
                    generator=torch.manual_seed(seed))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    # 구글 드라이브에 모델 저장
    drive_path = '/content/drive/MyDrive/3d_models'  # 구글 드라이브 내 저장 경로
    os.makedirs(drive_path, exist_ok=True)
    file_path = os.path.join(drive_path, f'mesh_{octree_resolution}.glb')
    mesh.export(file_path)

    # 3D 모델을 구글 드라이브 경로로 반환
    return file_path


def text_to_3d(prompt, num_steps, cfg_scale, seed, octree_resolution):
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = os.path.join(drive_model_folder, 'Hunyuan3D-2')  # 구글 드라이브 내 모델 경로 설정

    # 모델이 로드되지 않은 경우 구글 드라이브에서 다운로드
    if not os.path.exists(model_path):
        print("모델을 다운로드 중...")
        # 모델 다운로드 과정 (구글 드라이브에 모델 파일을 미리 업로드해야 합니다)
        # 모델 파일을 다운로드하는 코드가 필요하면 여기 추가

    # 텍스트로부터 이미지 생성
    image = t2i(prompt)

    # 이미지로 3D 모델 생성
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    mesh = i23d(image, num_inference_steps=num_steps, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    # 구글 드라이브에 모델 저장
    drive_path = '/content/drive/MyDrive/3d_models'  # 구글 드라이브 내 저장 경로
    os.makedirs(drive_path, exist_ok=True)
    file_path = os.path.join(drive_path, f't2i_{octree_resolution}.glb')
    mesh.export(file_path)

    # 3D 모델을 구글 드라이브 경로로 반환
    return file_path


def generate_3d_model(image, prompt, num_steps, cfg_scale, seed, octree_resolution, remove_bg):
    # 이미지와 텍스트 입력에 대해 3D 모델 생성
    mesh_image = image_to_3d(image, num_steps, cfg_scale, seed, octree_resolution, remove_bg)
    mesh_text = text_to_3d(prompt, num_steps, cfg_scale, seed, octree_resolution)

    return mesh_image, mesh_text


# Gradio 인터페이스 생성
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

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=True) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='Enter a text prompt to generate 3D model.',
                                             info='Example: A 3D model of a cute cat, white background')

                with gr.Accordion('Advanced Options', open=False):
                    num_steps = gr.Slider(maximum=50, minimum=20, value=50, step=1, label='Inference Steps')
                    octree_resolution = gr.Dropdown([256, 384, 512], value=256, label='Octree Resolution')
                    cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
                    seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')

                with gr.Group():
                    btn = gr.Button(value='Generate Shape Only', variant='primary')
                    btn_all = gr.Button(value='Generate Shape and Texture', variant='primary', visible=False)

                with gr.Group():
                    file_out = gr.File(label="Download White Mesh", interactive=True)
                    file_out2 = gr.File(label="Download Textured Mesh", interactive=True)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh') as mesh1:
                        html_output1 = gr.HTML("<div style='height: 596px; width: 100%; border-radius: 8px; border-color: #e5e7eb; order-style: solid; border-width: 1px;'></div>", label='Output')
                    with gr.Tab('Generated Textured Mesh') as mesh2:
                        html_output2 = gr.HTML("<div style='height: 596px; width: 100%; border-radius: 8px; border-color: #e5e7eb; order-style: solid; border-width: 1px;'></div>", label='Output')

            with gr.Column(scale=2):
                with gr.Tabs() as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery') as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=[], inputs=[image], label="Image Prompts", examples_per_page=18)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery') as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=[], inputs=[caption], label="Text Prompts", examples_per_page=18)

        tab_gi.select(fn=lambda: gr.update(selected='tab_img_prompt'), outputs=tabs_prompt)
        tab_gt.select(fn=lambda: gr.update(selected='tab_txt_prompt'), outputs=tabs_prompt)

        btn.click(
            generate_3d_model,
            inputs=[
                image,
                caption,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg
            ],
            outputs=[file_out, file_out2]
        )

    return demo


if __name__ == '__main__':
    demo = build_app()
    demo.queue(max_size=10)
    demo.launch(share=True)
