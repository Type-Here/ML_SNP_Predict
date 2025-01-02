import os
import subprocess

TF_VERSION = "2.18.0"
KERAS_VERSION = "3.7.0"
ROCM_VERSION = "6.3"

def install_tensorflow():
    try:
        # Verifica se esiste una GPU NVIDIA
        nvidia_check = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if nvidia_check.returncode == 0:
            print("NVIDIA GPU detected. Installing TensorFlow with CUDA support...")
            os.system(f"pip install tensorflow-gpu=={TF_VERSION}")
            os.system(f"pip install keras=={KERAS_VERSION}")
            return

        # Verifica se esiste una GPU AMD
        rocm_check = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rocm_check.returncode == 0:
            print("AMD GPU detected. Installing TensorFlow with ROCm support...")
            os.system(f"pip install tensorflow-rocm=={TF_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-{ROCM_VERSION}/")
            os.system(f"pip install keras=={KERAS_VERSION}")
            return

    except FileNotFoundError:
        pass

    # Nessuna GPU rilevata, installa TensorFlow per CPU
    print("No GPU detected. Installing TensorFlow for CPU...")
    os.system(f"pip install tensorflow=={TF_VERSION}")
    os.system(f"pip install keras=={KERAS_VERSION}")

if __name__ == "__main__":
    install_tensorflow()
