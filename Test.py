from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import torch

try:
    print("📥 모델 로드 시도...")
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    
    if i23d_worker is None:
        raise ValueError("❌ 모델 로드 실패: None 반환됨")

    i23d_worker = i23d_worker.to("cpu")
    print("✅ 모델 로드 성공:", type(i23d_worker))

except Exception as e:
    print("❌ 오류 발생:", e)
