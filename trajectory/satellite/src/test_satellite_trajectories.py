import math
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageStat
from shapely.geometry import Polygon

import generate_satellite_trajectories as pipeline


class SatellitePipelineTests(unittest.TestCase):
    def test_sampled_transitions_and_last_frame(self):
        turns, ratios, changes = [], [], []
        for seed in range(1000):
            frames, centers = pipeline.generate_local(random.Random(seed))
            self.assertEqual(len(frames), 8)
            for previous, current in zip(frames, frames[1:]):
                distance = math.hypot(current['center_x_m'] - previous['center_x_m'],
                                      current['center_y_m'] - previous['center_y_m'])
                ratio = distance / previous['extent_m']
                self.assertTrue(0.2 <= ratio <= 0.6)
                self.assertAlmostEqual(distance, previous['distance_to_next_m'])
                bearing = math.degrees(math.atan2(current['center_x_m'] - previous['center_x_m'],
                                                 current['center_y_m'] - previous['center_y_m'])) % 360
                self.assertAlmostEqual(bearing, previous['heading_deg'])
                self.assertTrue(pipeline.EXTENT_BOUNDS[0] <= current['extent_m'] <= pipeline.EXTENT_BOUNDS[1])
                self.assertLessEqual(abs(current['extent_change_from_previous_m']), 50)
                self.assertLessEqual(abs(current['turn_from_previous_deg']), 30)
                ratios.append(ratio)
                changes.append(current['extent_change_from_previous_m'])
            turns.extend(frame['turn_from_previous_deg'] for frame in frames[1:7])
            self.assertEqual(frames[-1]['heading_deg'], frames[-2]['heading_deg'])
            self.assertEqual(frames[-1]['turn_from_previous_deg'], 0)
            self.assertIsNone(frames[-1]['distance_to_next_m'])
            self.assertIsNone(frames[-1]['distance_extent_ratio'])
            if centers:
                self.assertEqual(len(centers), 8)
        self.assertAlmostEqual(sum(ratios) / len(ratios), 0.4, delta=0.005)
        self.assertAlmostEqual(sum(turns) / len(turns), 0, delta=1)
        self.assertFalse(any(value in (0.2, 0.6) for value in ratios))
        self.assertTrue(any(value < -20 for value in turns))
        self.assertTrue(any(value > 20 for value in turns))
        self.assertTrue(any(value < -40 for value in changes))
        self.assertTrue(any(value > 40 for value in changes))

    def test_zoom_is_minimum_adequate_source_resolution(self):
        for latitude in (0, 25, 45, 60):
            for extent in (100, 150, 300, 400):
                zoom = pipeline.choose_zoom(latitude, extent, 1024)
                self.assertGreaterEqual(extent / pipeline.tile_meters_per_pixel(latitude, zoom), 1024)
                self.assertLess(extent / pipeline.tile_meters_per_pixel(latitude, zoom - 1), 1024)
        self.assertGreater(pipeline.choose_zoom(0, 100, 1024), pipeline.choose_zoom(0, 400, 1024))
        self.assertGreater(pipeline.choose_zoom(0, 300, 1024), pipeline.choose_zoom(60, 300, 1024))
        self.assertEqual(pipeline.choose_zoom(0, 1, 1024, max_zoom=20), 20)

    def test_full_geographic_footprints_inside_concave_boundary(self):
        boundary = Polygon([(0, 0), (.05, 0), (.05, .05), (.025, .02), (0, .05)],
                           holes=[[(.008, .008), (.012, .008), (.012, .012), (.008, .012)]])
        for seed in range(20):
            lat, lon, frames = pipeline.place_trajectory(boundary, random.Random(seed))
            for frame in frames:
                self.assertTrue(boundary.covers(pipeline.geographic_footprint(lat, lon, frame)))

    def test_render_dynamic_zoom_and_coverage_fallback(self):
        class FakeStore:
            def __init__(self, available):
                self.available, self.calls = available, []

            def mosaic(self, lat, lon, zoom, side):
                self.calls.append(zoom)
                if zoom > self.available:
                    raise pipeline.TileCoverageError('missing')
                image = Image.new('RGB', (side + 32, side + 32), 'white')
                for x in range(0, image.width, 40):
                    image.paste((50, 120, 200), (x, 0, min(x + 20, image.width), image.height))
                return image, image.width / 2, image.height / 2

        with tempfile.TemporaryDirectory() as directory:
            for available in (22, 18):
                frame = {'latitude': 25, 'longitude': 10, 'extent_m': 120, 'heading_deg': 45}
                store = FakeStore(available)
                path = Path(directory) / f'{available}.jpg'
                pipeline.render_frame(store, frame, path, 1024)
                self.assertEqual(store.calls[0], frame['requested_zoom'])
                self.assertEqual(frame['zoom'], min(available, frame['requested_zoom']))
                self.assertEqual(frame['upsampled'], available == 18)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1024, 1024))
                    self.assertGreater(min(ImageStat.Stat(image).stddev), 5)
                    self.assertGreater(min(ImageStat.Stat(image).extrema[channel][0] for channel in range(3)), 0)
            with self.assertRaises(pipeline.TileCoverageError):
                pipeline.render_frame(FakeStore(17), frame, Path(directory) / 'missing.jpg', 1024)


if __name__ == '__main__':
    unittest.main()
