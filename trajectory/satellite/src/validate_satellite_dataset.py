"""Validate generated imagery, geometry, zoom and indexes without modifying data."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageStat
from shapely.geometry import LineString

import generate_satellite_trajectories as pipeline


def describe(values):
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': statistics.mean(values), 'stddev': statistics.pstdev(values)}


def validate(root, boundary_path, expected_total):
    boundaries = pipeline.load_boundaries(boundary_path)
    summary = json.loads((root / 'summary.json').read_text())
    catalog = json.loads((root / 'index.json').read_text())
    manifests = list(root.glob('cities/*/trajectory_sequences/*/manifest.json'))
    assert len(manifests) == expected_total, (len(manifests), expected_total)
    assert summary['total_trajectories'] == expected_total
    assert summary['total_images'] == expected_total * 8
    assert summary['cities'] == len(pipeline.CITY_IDS) == len(catalog)
    assert summary['pipeline_version'] == pipeline.PIPELINE_VERSION
    assert {row['city_id'] for row in catalog} == set(pipeline.CITY_IDS)
    ratios, distances, turns, changes, extents, image_deviations = [], [], [], [], [], []
    requested_zooms, actual_zooms, city_counts = Counter(), Counter(), Counter()
    fallback_count = upsample_count = 0
    for path in sorted(manifests):
        value = json.loads(path.read_text())
        frames, city = value['frames'], value['city_id']
        assert value['pipeline_version'] == pipeline.PIPELINE_VERSION, path
        assert value['trajectory_id'] == path.parent.name
        assert '__satellite_v4__' in path.parent.name
        assert value['num_control_points'] == 8
        assert len(frames) == 8 and value['gt_order'] == list(range(8))
        assert all(value['validation'][key] for key in ('extent_range_valid', 'extent_change_valid',
                                                      'distance_ratio_valid', 'heading_change_valid',
                                                      'boundary_footprints_valid'))
        assert not value['validation']['self_intersection']
        assert LineString([(frame['center_x_m'], frame['center_y_m']) for frame in frames]).is_simple
        city_counts[city] += 1
        for index, frame in enumerate(frames):
            assert frame['frame_id'] == index
            extent = frame['extent_m']
            assert pipeline.EXTENT_BOUNDS[0] <= extent <= pipeline.EXTENT_BOUNDS[1]
            extents.append(extent)
            assert boundaries[city].covers(pipeline.geographic_footprint(
                value['anchor']['latitude'], value['anchor']['longitude'], frame)), path
            lat, lon = pipeline.local_to_geo(value['anchor']['latitude'], value['anchor']['longitude'],
                                             frame['center_x_m'], frame['center_y_m'])
            assert abs(lat - frame['latitude']) < 1e-10 and abs(lon - frame['longitude']) < 1e-10
            requested = pipeline.choose_zoom(lat, extent, value['image_size'][0], value['max_zoom'])
            actual = frame['zoom']
            assert frame['requested_zoom'] == requested
            assert min(18, requested) <= actual <= requested
            assert math.isclose(frame['source_meters_per_pixel'], pipeline.tile_meters_per_pixel(lat, actual))
            assert math.isclose(frame['output_meters_per_pixel'], extent / value['image_size'][0])
            assert frame['upsampled'] == (frame['source_crop_pixels'] < value['image_size'][0])
            requested_zooms[requested] += 1
            actual_zooms[actual] += 1
            fallback_count += actual < requested
            upsample_count += frame['upsampled']
            with Image.open(path.parent / frame['file']) as image:
                assert image.size == tuple(value['image_size']) == (1024, 1024)
                image.load()
                deviation = max(ImageStat.Stat(image).stddev)
                assert deviation > 1, ('Near-uniform image', path, index, deviation)
                image_deviations.append(deviation)
            if index:
                previous = frames[index - 1]
                change = extent - previous['extent_m']
                signed_turn = (frame['heading_deg'] - previous['heading_deg'] + 180) % 360 - 180
                assert abs(change) <= 50 and abs(signed_turn) <= 30
                assert math.isclose(change, frame['extent_change_from_previous_m'], abs_tol=1e-9)
                assert math.isclose(signed_turn, frame['turn_from_previous_deg'], abs_tol=1e-9)
                changes.append(change)
                if index < 7:
                    turns.append(signed_turn)
            if index < 7:
                following = frames[index + 1]
                dx = following['center_x_m'] - frame['center_x_m']
                dy = following['center_y_m'] - frame['center_y_m']
                distance = math.hypot(dx, dy)
                ratio = distance / extent
                assert 0.2 <= ratio <= 0.6
                assert math.isclose(distance, frame['distance_to_next_m'])
                assert math.isclose(ratio, frame['distance_extent_ratio'])
                assert pipeline.turn(math.degrees(math.atan2(dx, dy)), frame['heading_deg']) < 1e-8
                shape_a, shape_b = pipeline.local_footprint(frame), pipeline.local_footprint(following)
                assert math.isclose(frame['overlap_with_next'], shape_a.intersection(shape_b).area / shape_a.area)
                ratios.append(ratio)
                distances.append(distance)
        assert frames[-1]['heading_deg'] == frames[-2]['heading_deg']
        assert frames[-1]['turn_from_previous_deg'] == 0
        assert frames[-1]['distance_to_next_m'] is None
        assert frames[-1]['distance_extent_ratio'] is None
        assert frames[-1]['overlap_with_next'] is None
        assert frames[0]['turn_from_previous_deg'] is None
        assert frames[0]['extent_change_from_previous_m'] is None
    for position, city in enumerate(pipeline.CITY_IDS):
        expected = expected_total // 30 + (position < expected_total % 30)
        city_root = root / 'cities' / city
        rows = json.loads((city_root / 'index.json').read_text())
        city_summary = json.loads((city_root / 'summary.json').read_text())
        assert city_counts[city] == len(rows) == city_summary['complete'] == expected
        assert city_summary['image_count'] == expected * 8
        assert next(row['complete'] for row in catalog if row['city_id'] == city) == expected
        assert {row['id'] for row in rows} == {f'{city}__satellite_v4__{index:06d}' for index in range(expected)}
        for row in rows:
            assert (city_root / row['path']).is_file()
    assert len(list(root.glob('cities/*/trajectory_sequences/*/frame_*.jpg'))) == expected_total * 8
    return {'valid': True, 'pipeline_version': pipeline.PIPELINE_VERSION, 'seed': summary['seed'],
            'trajectories': expected_total, 'images': expected_total * 8, 'cities': len(city_counts),
            'city_counts': dict(city_counts), 'requested_zoom_counts': dict(sorted(requested_zooms.items())),
            'actual_zoom_counts': dict(sorted(actual_zooms.items())), 'fallback_frames': fallback_count,
            'upsampled_frames': upsample_count, 'distance_ratio': describe(ratios),
            'distance_m': describe(distances), 'heading_turn_deg_excluding_fixed_last': describe(turns),
            'extent_change_m': describe(changes), 'extent_m': describe(extents),
            'minimum_image_channel_stddev': min(image_deviations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--boundaries', type=Path, required=True)
    parser.add_argument('--expected-total', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_root, args.boundaries, args.expected_total), indent=2))


if __name__ == '__main__':
    main()
