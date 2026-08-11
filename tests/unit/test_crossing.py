"""Pruebas unitarias de la máquina de estados de cruce entre zonas."""

from app.vision.crossing import CrossingEvent, TrackZoneState


STABLE_FRAMES = 2


def confirm_zone(
    state: TrackZoneState,
    zone: str,
    started_at: float,
) -> CrossingEvent | None:
    """
    Simula que una persona permanece el número mínimo de fotogramas
    necesario para confirmar una zona.
    """
    detected_event: CrossingEvent | None = None

    for offset in range(STABLE_FRAMES):
        current_event = state.update(
            detected_zone=zone,
            now=started_at + offset * 0.01,
            minimum_stable_frames=STABLE_FRAMES,
            event_cooldown_seconds=0.0,
        )

        if current_event is not None:
            detected_event = current_event

    return detected_event


def test_detects_entry() -> None:
    """Detecta entrada en la secuencia exterior, puerta e interior."""

    state = TrackZoneState()

    assert confirm_zone(state, "exterior", 0.0) is None
    assert confirm_zone(state, "door", 1.0) is None

    event = confirm_zone(state, "interior", 2.0)

    assert event == CrossingEvent.ENTRY


def test_detects_exit() -> None:
    """Detecta salida en la secuencia interior, puerta y exterior."""

    state = TrackZoneState()

    assert confirm_zone(state, "interior", 0.0) is None
    assert confirm_zone(state, "door", 1.0) is None

    event = confirm_zone(state, "exterior", 2.0)

    assert event == CrossingEvent.EXIT


def test_does_not_register_approach_as_entry() -> None:
    """Rechaza una aproximación a la puerta que retorna al exterior."""

    state = TrackZoneState()

    assert confirm_zone(state, "exterior", 0.0) is None
    assert confirm_zone(state, "door", 1.0) is None

    event = confirm_zone(state, "exterior", 2.0)

    assert event is None


def test_does_not_register_movement_inside_same_zone() -> None:
    """Ignora permanencias repetidas dentro de una misma zona."""

    state = TrackZoneState()

    assert confirm_zone(state, "interior", 0.0) is None

    event = confirm_zone(state, "interior", 1.0)

    assert event is None


def test_entry_event_is_not_repeated() -> None:
    """Impide repetir la entrada mientras no exista un nuevo cruce."""

    state = TrackZoneState()

    confirm_zone(state, "exterior", 0.0)
    confirm_zone(state, "door", 1.0)

    first_event = confirm_zone(state, "interior", 2.0)
    repeated_event = confirm_zone(state, "interior", 3.0)

    assert first_event == CrossingEvent.ENTRY
    assert repeated_event is None
