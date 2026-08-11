from __future__ import annotations

import platform
import sys

import cv2
import torch


def bytes_to_gb(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    print("=" * 60)
    print("VERIFICACIÓN DEL ENTORNO")
    print("=" * 60)

    print(f"Python:          {sys.version.split()[0]}")
    print(f"Sistema:         {platform.system()} {platform.release()}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"OpenCV:          {cv2.__version__}")
    print(f"CUDA en wheel:   {torch.version.cuda}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detectó CUDA. Revisa el controlador NVIDIA "
            "y confirma que instalaste la compilación cu132."
        )

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)

    print(f"GPU:             {properties.name}")
    print(f"Capacidad CUDA:  {properties.major}.{properties.minor}")
    print(f"VRAM total:      {bytes_to_gb(properties.total_memory):.2f} GB")
    print(f"cuDNN:           {torch.backends.cudnn.version()}")

    device = torch.device("cuda:0")

    print("\nEjecutando una operación de prueba en GPU...")

    matrix_a = torch.randn((1024, 1024), device=device)
    matrix_b = torch.randn((1024, 1024), device=device)
    result = matrix_a @ matrix_b

    torch.cuda.synchronize()

    print(f"Tensor resultante: {tuple(result.shape)}")
    print(f"Dispositivo:       {result.device}")
    print(f"Promedio:          {result.mean().item():.6f}")

    print("\nENTORNO CONFIGURADO CORRECTAMENTE")


if __name__ == "__main__":
    main()