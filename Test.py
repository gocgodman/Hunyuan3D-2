from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import torch
from huggingface_hub import from_pretrained 

try:
    print("📥 모델 로드 시도...")
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2").to("cpu")
    
    # 모델 로드 후, None 체크
    if not i23d_worker:
        raise ValueError("❌ 모델 로드 실패: None 반환됨")
    
    print("✅ 모델 로드 성공:", type(i23d_worker))

except Exception as e:
    print("❌ 오류 발생:", e)
