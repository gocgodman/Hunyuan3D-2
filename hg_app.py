import argparse
import os
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
from huggingface_hub import login
import torch  # 🔹 PyTorch 추가 (CPU 모드로 설정)

# 자신의 허깅페이스 토큰을 입력하세요
login(token="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")

# **CLI 인자 설정**
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--cache-path', type=str, default='gradio_cache')
parser.add_argument('--enable_t23d', default=True)
parser.add_argument('--local', action="store_true")
args = parser.parse_args()

# **GPU 대신 CPU 강제 사용**
device = "cpu"  # 🔹 GPU 없이 CPU로만 실행
print(f"🔥 실행 장치: {device}")

# 서버 설정
IP = "0.0.0.0"
PORT = 7860 if not args.local else 8080

# **3D 모델 변환 함수**
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def generation_all(
    caption: str,
    image: Image.Image,
    steps: int,
    guidance_scale: float,
    seed: int,
    octree_resolution: int,
):
    print("🚀 3D 모델 생성 시작")
    
    # 🔹 CPU에서 실행하도록 변경
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        torch_dtype=torch.float32  # ✅ CPU 모드에서 실행 가능하도록 수정
    ).to("cpu")  # ✅ 명확하게 CPU로 설정


    save_folder = os.path.join(SAVE_DIR, "output")
    os.makedirs(save_folder, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(int(seed))  # 🔹 CPU 전용 Generator 사용

    # 3D 모델 생성
    mesh = i23d_worker.forward(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    # 저장 경로
    white_mesh_path = os.path.join(save_folder, "white_mesh.glb")
    mesh.export(white_mesh_path, include_normals=False)

    # 텍스처 생성 (CPU 모드)
    print("🎨 텍스처 생성 중...")
    texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        torch_dtype=torch.float32  # ✅ CPU 모드에서 실행 가능하도록 수정
    ).to("cpu")  # ✅ 명확하게 CPU로 설정


    textured_mesh = texgen_worker(mesh, image)
    textured_mesh_path = os.path.join(save_folder, "textured_mesh.glb")
    textured_mesh.export(textured_mesh_path, include_normals=True)

    print("✅ 3D 모델 생성 완료")
    return white_mesh_path, textured_mesh_path

# **FastAPI 및 Gradio 설정**
app = FastAPI()
static_dir = Path(args.cache_path)
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# **Gradio 인터페이스 추가**
with gr.Blocks(title="Hunyuan3D-2.0") as demo:
    gr.Markdown("# 🏗️ Hunyuan3D-2.0")
    gr.Markdown("### 이미지를 업로드하고 3D 모델을 생성하세요!")

    with gr.Row():
        image_input = gr.Image(type="pil", label="이미지 업로드")
        caption_input = gr.Textbox(label="설명 (선택 사항)", placeholder="이미지 설명 입력...")

    with gr.Row():
        steps_input = gr.Slider(10, 50, value=30, step=1, label="추론 단계")
        guidance_input = gr.Slider(1.0, 10.0, value=7.5, step=0.1, label="Guidance Scale")
        seed_input = gr.Number(value=42, label="랜덤 시드")
        octree_input = gr.Slider(128, 1024, value=512, step=128, label="Octree 해상도")

    output_1 = gr.File(label="흰색 3D 모델 (GLB)")
    output_2 = gr.File(label="텍스처 3D 모델 (GLB)")

    generate_btn = gr.Button("🚀 3D 모델 생성")

    generate_btn.click(
        fn=generation_all,
        inputs=[caption_input, image_input, steps_input, guidance_input, seed_input, octree_input],
        outputs=[output_1, output_2],
    )

app = gr.mount_gradio_app(app, demo, path="/")

# 서버 실행
if __name__ == '__main__':
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(CURRENT_DIR, args.cache_path)
    os.makedirs(SAVE_DIR, exist_ok=True)

    uvicorn.run(app, host=IP, port=PORT)
