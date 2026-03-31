from picamera2 import Picamera2


class FrameSource:
    """Camera-based frame source using Picamera2."""

    def __init__(
        self,
        path: str = None,
        loop: bool = True,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ):
        self._loop = loop
        self._fps = fps

        self._camera = Picamera2()
        config = self._camera.create_preview_configuration(
            main={"size": (width, height)},
            lores={"size": (width, height)}
        )
        self._camera.configure(config)
        self._camera.start()

    # ---- properties ------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        config = self._camera.camera_configuration()
        return config["main"]["size"][0]

    @property
    def height(self) -> int:
        config = self._camera.camera_configuration()
        return config["main"]["size"][1]

    @property
    def frame_count(self) -> int:
        return -1  # live camera = infinite

    # ---- core API --------------------------------------------------

    def read(self):
        frame = self._camera.capture_array()
        return frame

    def release(self):
        self._camera.stop()

    # ---- context manager -------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __repr__(self):
        return (
            f"FrameSource(camera, "
            f"{self.width}x{self.height} @ {self.fps:.1f} fps)"
        )