import os
import subprocess
import sys

# 1️⃣ Python 버전 및 아키텍처 확인
python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
architecture = os.popen("uname -m").read().strip()

print(f"📌 Python 버전: {sys.version.split()[0]}")
print(f"📌 아키텍처: {architecture}")

# 2️⃣ 필수 패키지 업데이트
print("\n🔄 pip 및 빌드 도구 업데이트...")
subprocess.run(["pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

# 3️⃣ 올바른 휠 파일 이름 결정
if architecture == "x86_64":
    whl_file = f"custom_rasterizer-0.1-{python_version}-{python_version}-manylinux_x86_64.whl"
elif architecture == "aarch64":
    whl_file = f"custom_rasterizer-0.1-{python_version}-{python_version}-manylinux_aarch64.whl"
else:
    whl_file = None  # 지원되지 않는 아키텍처

# 4️⃣ 휠 설치 또는 소스 빌드
try:
    if whl_file:
        print(f"\n🚀 휠 파일 설치 시도: {whl_file}")
        subprocess.run(["pip", "install", whl_file], check=True)
    else:
        raise FileNotFoundError("지원되는 휠 파일이 없음")

except (subprocess.CalledProcessError, FileNotFoundError):
    print("\n⚠️ 휠 파일이 지원되지 않음. 소스에서 직접 빌드하여 설치합니다...")
    subprocess.run(["pip", "install", "--no-binary", ":all:", "custom_rasterizer"], check=True)

print("\n✅ 설치 완료!")
