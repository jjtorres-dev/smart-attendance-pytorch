from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


ZONE_NAMES: Final[tuple[str, ...]] = (
    "exterior",
    "door",
    "interior",
)

# La puerta se evalúa primero para que tenga prioridad
# cuando dos polígonos comparten algún borde.
ZONE_PRIORITY: Final[tuple[str, ...]] = (
    "door",
    "interior",
    "exterior",
)

ZONE_LABELS: Final[dict[str, str]] = {
    "exterior": "EXTERIOR",
    "door": "PUERTA",
    "interior": "INTERIOR",
}

ZONE_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "exterior": (0, 165, 255),
    "door": (0, 255, 255),
    "interior": (0, 200, 0),
}


@dataclass(frozen=True)
class ZoneDefinition:
    """
    Polígono almacenado usando coordenadas normalizadas entre 0 y 1.
    """

    name: str
    normalized_points: tuple[tuple[float, float], ...]

    def to_pixel_points(
        self,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        points = [
            (
                int(x * frame_width),
                int(y * frame_height),
            )
            for x, y in self.normalized_points
        ]

        return np.array(points, dtype=np.int32)


class ZoneMap:
    """
    Contiene las zonas de exterior, puerta e interior.
    """

    def __init__(
        self,
        zones: dict[str, ZoneDefinition],
    ) -> None:
        missing_zones = set(ZONE_NAMES) - set(zones)

        if missing_zones:
            raise ValueError(
                "Faltan zonas obligatorias: "
                + ", ".join(sorted(missing_zones))
            )

        self._zones = zones

    @classmethod
    def load(cls, path: str | Path) -> "ZoneMap":
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No se encontró la configuración de zonas: {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        raw_zones = data.get("zones")

        if not isinstance(raw_zones, dict):
            raise ValueError(
                "El archivo de zonas no contiene la sección 'zones'."
            )

        zones: dict[str, ZoneDefinition] = {}

        for zone_name in ZONE_NAMES:
            raw_points = raw_zones.get(zone_name)

            if not isinstance(raw_points, list):
                raise ValueError(
                    f"La zona {zone_name!r} no contiene puntos válidos."
                )

            if len(raw_points) < 3:
                raise ValueError(
                    f"La zona {zone_name!r} debe tener al menos 3 puntos."
                )

            normalized_points: list[tuple[float, float]] = []

            for point in raw_points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(
                        f"Punto inválido en la zona {zone_name!r}: {point!r}"
                    )

                x = float(point[0])
                y = float(point[1])

                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise ValueError(
                        f"Las coordenadas deben estar entre 0 y 1: {point!r}"
                    )

                normalized_points.append((x, y))

            zones[zone_name] = ZoneDefinition(
                name=zone_name,
                normalized_points=tuple(normalized_points),
            )

        return cls(zones)

    def polygon(
        self,
        zone_name: str,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray:
        frame_height, frame_width = frame_shape[:2]

        return self._zones[zone_name].to_pixel_points(
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def zone_at(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> str | None:
        """
        Devuelve la zona que contiene el punto.

        Si el punto no está dentro de ninguna zona, devuelve None.
        """
        point_as_float = (
            float(point[0]),
            float(point[1]),
        )

        for zone_name in ZONE_PRIORITY:
            polygon = self.polygon(
                zone_name=zone_name,
                frame_shape=frame_shape,
            )

            result = cv2.pointPolygonTest(
                polygon,
                point_as_float,
                False,
            )

            if result >= 0:
                return zone_name

        return None

    def draw(
        self,
        frame: np.ndarray,
        opacity: float = 0.18,
    ) -> np.ndarray:
        """
        Dibuja las zonas sobre el fotograma.
        """
        overlay = frame.copy()

        for zone_name in ZONE_NAMES:
            polygon = self.polygon(
                zone_name=zone_name,
                frame_shape=frame.shape,
            )

            color = ZONE_COLORS[zone_name]

            cv2.fillPoly(
                overlay,
                [polygon],
                color,
            )

        cv2.addWeighted(
            overlay,
            opacity,
            frame,
            1.0 - opacity,
            0,
            frame,
        )

        for zone_name in ZONE_NAMES:
            polygon = self.polygon(
                zone_name=zone_name,
                frame_shape=frame.shape,
            )

            color = ZONE_COLORS[zone_name]

            cv2.polylines(
                frame,
                [polygon],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            center_x = int(np.mean(polygon[:, 0]))
            center_y = int(np.mean(polygon[:, 1]))

            cv2.putText(
                frame,
                ZONE_LABELS[zone_name],
                (center_x - 50, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        return frame