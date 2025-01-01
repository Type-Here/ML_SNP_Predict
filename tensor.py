import os
import subprocess

def install_tensorflow():
    try:
        # Verifica se esiste una GPU NVIDIA
        nvidia_check = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if nvidia_check.returncode == 0:
            print("NVIDIA GPU detected. Installing TensorFlow with CUDA support...")
            os.system("pip install tensorflow==2.17.1")
            return

        # Verifica se esiste una GPU AMD
        rocm_check = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rocm_check.returncode == 0:
            print("AMD GPU detected. Installing TensorFlow with ROCm support...")
            rocm_version = "6.3"  # Cambia con la versione ROCm installata
            os.system(f"pip install tensorflow-rocm==2.17.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm_version}/")
            return

    except FileNotFoundError:
        pass

    # Nessuna GPU rilevata, installa TensorFlow per CPU
    print("No GPU detected. Installing TensorFlow for CPU...")
    os.system("pip install tensorflow==2.17.1 --no-deps")

if __name__ == "__main__":
    install_tensorflow()
