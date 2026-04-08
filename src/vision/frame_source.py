import cv2 as cv

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


class FrameSource:
    """Unified frame source: video file (OpenCV) or Pi camera (Picamera2).

    If *path* is provided and is a valid file, OpenCV VideoCapture is used.
    Otherwise, Picamera2 is used (requires running on a Raspberry Pi).
    """

    def __init__(
        self,
        path=None,
        loop: bool = True,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ):
        self._loop = loop
        self._fps = fps
        self._width = width
        self._height = height
        self._use_camera = False

        # Decide backend: OpenCV source (camera index or file path) or Picamera2.
        source = None
        if isinstance(path, int):
            source = path
        elif isinstance(path, str):
            value = path.strip()
            if value != "":
                source = int(value) if value.isdigit() else value

        if source is not None:
            self._cap = cv.VideoCapture(source)
            if not self._cap.isOpened():
                raise FileNotFoundError(
                    f"Cannot open video source: {path}"
                )
            self._fps = self._cap.get(cv.CAP_PROP_FPS) or fps
            self._width = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self._frame_count = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))
            self._source_label = str(path)
        else:
            # Live camera
            if not _HAS_PICAMERA2:
                raise ImportError(
                    "picamera2 is not installed and no video file was provided. "
                    "Install picamera2 (on Raspberry Pi) or set input.video_path "
                    "in vision_config.yaml."
                )
            self._use_camera = True
            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": (width, height)},
                lores={"size": (width, height)},
            )
            self._camera.configure(config)
            self._camera.start()
            self._frame_count = -1  # live = infinite
            self._source_label = "picamera2"

    # ---- properties ------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        if self._use_camera:
            cfg = self._camera.camera_configuration()
            return cfg["main"]["size"][0]
        return self._width

    @property
    def height(self) -> int:
        if self._use_camera:
            cfg = self._camera.camera_configuration()
            return cfg["main"]["size"][1]
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ---- core API --------------------------------------------------

    def read(self):
        if self._use_camera:
            return self._camera.capture_array()

        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                return None
        return frame

    def release(self):
        if self._use_camera:
            self._camera.stop()
        else:
            self._cap.release()

    # ---- context manager -------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __repr__(self):
        return (
            f"FrameSource({self._source_label}, "
            f"{self.width}x{self.height} @ {self.fps:.1f} fps)"
        )