"""Opt-in GPU/desktop smoke: HIL_TEST_PREVIEW=1 python -m unittest ...

Opens one temporary preview window. Does not connect to a policy or run HIL.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
import unittest


@unittest.skipUnless(os.environ.get("HIL_TEST_PREVIEW") == "1", "requires explicit GPU/desktop opt-in")
class PreviewIntegrationTest(unittest.TestCase):
    def test_small_framebuffer_large_window_and_hil_keys(self):
        import numpy as np
        import sapien

        path = Path(__file__).resolve().parents[2] / "envs" / "hil_preview.py"
        spec = importlib.util.spec_from_file_location("hil_preview", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)
        scene = engine.create_scene()
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_ground(0)
        builder = scene.create_actor_builder()
        material = renderer.create_material()
        material.base_color = [1, 0, 0, 1]
        builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=material)
        box = builder.build_static(name="preview_smoke_box")
        box.set_pose(sapien.Pose([0, 0, 0.1]))
        policy_camera = scene.add_camera("policy_camera", 64, 48, 1.0, 0.1, 10)
        viewer = module.LowResolutionViewer((320, 180), (1600, 900))
        try:
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1, 0, 1)
            viewer.set_camera_rpy(0, -0.8, np.pi)
            for _ in range(3):
                viewer.render()
            self.assertEqual(viewer.last_image_shape[:2], (180, 320))
            self.assertEqual(tuple(viewer.size), (1600, 900))
            image = viewer.camera.get_picture("Color")
            self.assertGreater(float(image[..., :3].std()), 0.01)

            # A larger desktop window must not resize either camera.
            viewer.resize(1920, 1080)
            for _ in range(3):
                viewer.render()
            self.assertEqual(tuple(viewer.size), (1920, 1080))
            self.assertEqual(viewer.last_image_shape[:2], (180, 320))
            self.assertEqual((policy_camera.width, policy_camera.height), (64, 48))
            viewer.resize(1600, 900)

            # Repeat KEYDOWN events cannot repeatedly toggle enhanced sampling.
            pg = viewer._pg
            for key in ("e", "i", "r", "x"):
                code = pg.key.key_code(key)
                pg.event.post(pg.event.Event(pg.KEYDOWN, key=code))
                viewer.render()
                self.assertTrue(viewer.key_press(key))
                self.assertTrue(viewer.key_down(key))
                pg.event.post(pg.event.Event(pg.KEYDOWN, key=code))
                viewer.render()
                self.assertFalse(viewer.key_press(key))
                pg.event.post(pg.event.Event(pg.KEYUP, key=code))
                viewer.render()
                self.assertFalse(viewer.key_down(key))

            start = time.perf_counter()
            for _ in range(20):
                viewer.render()
            print(json.dumps({
                "preview_shape": viewer.last_image_shape,
                "window_size": viewer.size,
                "policy_camera_size": [policy_camera.width, policy_camera.height],
                "simple_scene_mean_render_ms": (time.perf_counter() - start) * 1000 / 20,
                "note": "renderer smoke only; not full task/control throughput",
            }), flush=True)
            pg.event.post(pg.event.Event(pg.QUIT))
            viewer.render()
            self.assertTrue(viewer.key_press("q"))
            self.assertTrue(viewer.closed)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
