"""Calibración interactiva de polígonos para exterior, puerta e interior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZONE_ORDER = (
    "exterior",
    "door",
    "interior",
)

ZONE_LABELS = {
    "exterior": "EXTERIOR",
    "door": "PUERTA",
    "interior": "INTERIOR",
}

ZONE_COLORS = {
    "exterior": (0, 165, 255),
    "door": (0, 255, 255),
    "interior": (0, 200, 0),
}


def parse_source(value: str) -> int | str:
    """Interpreta una fuente numérica como cámara y conserva otras rutas."""

    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def capture_reference_frame(
    source: int | str,
) -> np.ndarray:
    """Abre la fuente y obtiene un fotograma estable para la calibración."""

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara: {source!r}"
        )

    frame: np.ndarray | None = None

    # Permitimos que la transmisión se estabilice.
    for _ in range(30):
        success, current_frame = capture.read()

        if success and current_frame is not None:
            frame = current_frame

        time.sleep(0.02)

    capture.release()

    if frame is None:
        raise RuntimeError(
            "No se pudo obtener un fotograma de referencia."
        )

    return frame


def save_configuration(
    points: dict[str, list[tuple[int, int]]],
    frame: np.ndarray,
    output_path: Path,
    reference_path: Path,
) -> None:
    """Normaliza los puntos y guarda la configuración y su referencia visual."""

    frame_height, frame_width = frame.shape[:2]

    normalized_zones: dict[str, list[list[float]]] = {}

    for zone_name, zone_points in points.items():
        # Las coordenadas relativas independizan la calibración de la
        # resolución utilizada posteriormente durante la captura.
        normalized_zones[zone_name] = [
            [
                round(x / frame_width, 6),
                round(y / frame_height, 6),
            ]
            for x, y in zone_points
        ]

    configuration = {
        "version": 1,
        "reference_resolution": {
            "width": frame_width,
            "height": frame_height,
        },
        "zones": normalized_zones,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            ensure_ascii=False,
            indent=2,
        )

    cv2.imwrite(
        str(reference_path),
        frame,
    )

    print(f"Configuración guardada: {output_path}")
    print(f"Imagen de referencia:    {reference_path}")


def draw_interface(
    frame: np.ndarray,
    points: dict[str, list[tuple[int, int]]],
    current_zone_index: int,
) -> np.ndarray:
    """Dibuja polígonos, vértices e instrucciones sobre el fotograma."""

    canvas = frame.copy()
    overlay = canvas.copy()

    for zone_name in ZONE_ORDER:
        zone_points = points[zone_name]

        if len(zone_points) < 1:
            continue

        polygon = np.array(
            zone_points,
            dtype=np.int32,
        )

        color = ZONE_COLORS[zone_name]

        if len(zone_points) >= 3:
            cv2.fillPoly(
                overlay,
                [polygon],
                color,
            )

        if len(zone_points) >= 2:
            cv2.polylines(
                canvas,
                [polygon],
                isClosed=len(zone_points) >= 3,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        for point in zone_points:
            cv2.circle(
                canvas,
                point,
                6,
                color,
                -1,
            )

    cv2.addWeighted(
        overlay,
        0.18,
        canvas,
        0.82,
        0,
        canvas,
    )

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 135),
        (0, 0, 0),
        -1,
    )

    if current_zone_index < len(ZONE_ORDER):
        current_zone = ZONE_ORDER[current_zone_index]
        current_text = (
            f"Dibujando: {ZONE_LABELS[current_zone]}"
        )
    else:
        current_text = "Todas las zonas están dibujadas"

    instructions = [
        current_text,
        "Clic izquierdo: agregar punto | Clic derecho: deshacer",
        "ENTER: terminar zona | R: reiniciar zona | C: reiniciar todo",
        "S: guardar | Q o ESC: cancelar",
    ]

    for index, text in enumerate(instructions):
        cv2.putText(
            canvas,
            text,
            (15, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def run_calibration(args: argparse.Namespace) -> None:
    """Gestiona la interacción de ratón y teclado para definir las zonas."""

    source = parse_source(args.source)

    print("Capturando imagen de referencia...")
    frame = capture_reference_frame(source)

    points: dict[str, list[tuple[int, int]]] = {
        zone_name: []
        for zone_name in ZONE_ORDER
    }

    state = {
        "current_zone_index": 0,
    }

    window_name = "Smart Attendance - Calibración de zonas"

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        """Añade o elimina vértices de la zona que se está editando."""

        del flags, parameter

        current_index = state["current_zone_index"]

        if current_index >= len(ZONE_ORDER):
            return

        current_zone = ZONE_ORDER[current_index]

        if event == cv2.EVENT_LBUTTONDOWN:
            points[current_zone].append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if points[current_zone]:
                points[current_zone].pop()

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    output_path = PROJECT_ROOT / args.output
    reference_path = PROJECT_ROOT / args.reference

    try:
        while True:
            canvas = draw_interface(
                frame=frame,
                points=points,
                current_zone_index=state["current_zone_index"],
            )

            cv2.imshow(
                window_name,
                canvas,
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                print("Calibración cancelada.")
                break

            if key == ord("r"):
                current_index = state["current_zone_index"]

                if current_index < len(ZONE_ORDER):
                    current_zone = ZONE_ORDER[current_index]
                    points[current_zone].clear()

            if key == ord("c"):
                for zone_points in points.values():
                    zone_points.clear()

                state["current_zone_index"] = 0

            if key in (13, 10):
                current_index = state["current_zone_index"]

                if current_index >= len(ZONE_ORDER):
                    continue

                current_zone = ZONE_ORDER[current_index]

                if len(points[current_zone]) < 3:
                    print(
                        f"La zona {current_zone} necesita "
                        "al menos 3 puntos."
                    )
                    continue

                state["current_zone_index"] += 1

            if key == ord("s"):
                # La persistencia solo se habilita si cada polígono contiene
                # el mínimo geométrico de tres vértices.
                incomplete_zones = [
                    zone_name
                    for zone_name in ZONE_ORDER
                    if len(points[zone_name]) < 3
                ]

                if incomplete_zones:
                    print(
                        "Todavía faltan zonas: "
                        + ", ".join(incomplete_zones)
                    )
                    continue

                save_configuration(
                    points=points,
                    frame=frame,
                    output_path=output_path,
                    reference_path=reference_path,
                )

                break

    finally:
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos para la calibración."""

    parser = argparse.ArgumentParser(
        description="Calibra las zonas de entrada y salida."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Índice de cámara, URL o archivo de video.",
    )

    parser.add_argument(
        "--output",
        default="config/zones.json",
        help="Archivo JSON donde se guardarán las zonas.",
    )

    parser.add_argument(
        "--reference",
        default="config/zones_reference.jpg",
        help="Imagen tomada durante la calibración.",
    )

    return parser


def main() -> None:
    """Procesa los argumentos e inicia la interfaz interactiva."""

    parser = build_parser()
    arguments = parser.parse_args()
    run_calibration(arguments)


if __name__ == "__main__":
    main()
