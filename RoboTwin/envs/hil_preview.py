"""Low-resolution spectator camera displayed in an independently sized SDL window.

SAPIEN RenderWindow couples its scene framebuffer to the native window size.
This HIL-only viewer uses a separate camera and SDL's scaled presentation so
enlarging the window never enlarges the 3-D render target or policy cameras.
"""

from __future__ import annotations

import os

import numpy as np
import sapien
from sapien.utils.viewer.camera_control import FPSCameraController


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Expected a positive WIDTHxHEIGHT resolution, got {value!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {value!r}")
    return width, height


class LowResolutionViewer:
    """Small Viewer-compatible surface for the interactive HG-DAgger workflow."""

    manages_fullscreen = True
    plugins = ()
    paused = False

    def __init__(self, resolutions=(320, 180), window_size=(1600, 900), fullscreen=False):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from pygame._sdl2.video import Window

        self._pg = pygame
        self._resolution = tuple(resolutions)
        self._fullscreen = bool(fullscreen)
        self._closed = False
        self._pressed = set()
        self._down = set()
        self._reported = False
        self._controller = FPSCameraController()
        self.scene = None
        self.camera = None
        self._camera_entity = None
        self.last_image_shape = None
        self.window = self  # key_press/key_down/size facade used by KeyReader
        self.shader_dir = sapien.render.get_viewer_shader_dir()

        pygame.display.init()
        flags = pygame.SCALED | pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        # SDL scales this small surface during presentation. No 1600x900 scene
        # image is rendered, read back, or resized in Python.
        self._surface = pygame.display.set_mode(self._resolution, flags, vsync=0)
        pygame.display.set_caption("RoboTwin HIL — low-resolution preview")
        pygame.key.set_repeat()  # held e must not repeatedly toggle sampling
        self._native_window = Window.from_display_module()
        if not fullscreen:
            self._native_window.size = tuple(window_size)

    @property
    def resolution(self):
        return self._resolution

    @property
    def size(self):
        return self._pg.display.get_window_size() if not self.closed else (0, 0)

    @property
    def closed(self):
        return self._closed

    @property
    def should_close(self):
        return self._closed

    def resize(self, width, height):
        """Resize only the desktop window; the spectator camera stays fixed."""
        self._native_window.size = (int(width), int(height))

    def set_scene(self, scene):
        if self.scene is not None and self._camera_entity is not None:
            self.scene.remove_entity(self._camera_entity)
        self.scene = scene
        self.camera = sapien.render.RenderCameraComponent(*self._resolution, self.shader_dir)
        self.camera.near = 0.1
        self.camera.far = 1000.0
        self.camera.set_fovy(np.pi / 2, compute_x=True)
        self._camera_entity = sapien.Entity()
        self._camera_entity.name = "_hil_spectator_camera"
        self._camera_entity.add_component(self.camera)
        self._camera_entity.pose = self._controller.pose
        scene.add_entity(self._camera_entity)

    def set_camera_xyz(self, x, y, z):
        self._controller.setXYZ(x, y, z)
        self._camera_entity.pose = self._controller.pose

    def set_camera_rpy(self, r, p, y):
        self._controller.setRPY(r, p, y)
        self._camera_entity.pose = self._controller.pose

    def _poll_events(self):
        pg = self._pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # Route the close button through the existing graceful q path.
                self._pressed.add("q")
                self._closed = True
            elif event.type == pg.KEYDOWN:
                key = pg.key.name(event.key).lower()
                if key not in self._down:
                    self._pressed.add(key)
                self._down.add(key)
            elif event.type == pg.KEYUP:
                self._down.discard(pg.key.name(event.key).lower())
            elif event.type == pg.WINDOWFOCUSLOST:
                self._down.clear()
            elif event.type == pg.MOUSEMOTION and event.buttons[2]:
                dx, dy = event.rel
                self._controller.rotate(0, -dy * 0.005, dx * 0.005)
                self._camera_entity.pose = self._controller.pose
            elif event.type == pg.MOUSEWHEEL:
                self._controller.move(event.y * 0.05, 0, 0)
                self._camera_entity.pose = self._controller.pose

    def key_press(self, key):
        if key in self._pressed:
            self._pressed.remove(key)
            return True
        return False

    def key_down(self, key):
        return key in self._down

    def get_picture_size(self, name="Color"):
        return self._resolution

    def render(self):
        if self.closed:
            return
        self._poll_events()
        if self.closed or self.scene is None:
            return
        self.scene.update_render()
        self.camera.take_picture()
        image = self.camera.get_picture("Color")
        self.last_image_shape = tuple(image.shape)
        width, height = self._resolution
        if image.shape[:2] != (height, width):
            raise RuntimeError(f"Preview framebuffer drifted: {image.shape}, expected {(height, width)}")
        rgb = np.clip(image[..., :3] * 255, 0, 255).astype(np.uint8)
        self._pg.surfarray.blit_array(self._surface, rgb.transpose(1, 0, 2))
        self._pg.display.flip()
        if not self._reported:
            print(
                f"[HIL-VIEWER] backend=preview rendered={width}x{height} "
                f"window={self.size[0]}x{self.size[1]} fullscreen={self._fullscreen}; "
                "e/i/r/x/q enabled, right-drag=look, wheel=move camera",
                flush=True,
            )
            self._reported = True

    def close(self):
        self._closed = True
        if self.scene is not None and self._camera_entity is not None:
            self.scene.remove_entity(self._camera_entity)
        self._camera_entity = None
        self.camera = None
        self.scene = None
        self._pg.display.quit()
