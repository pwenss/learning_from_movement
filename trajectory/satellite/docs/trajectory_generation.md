# Satellite Imagery Trajectory Generation Pipeline

Version: `satellite_v4_extent_100_400_truncated_normal_dynamic_zoom`

## 1. Dataset and Frame Definition

Generate independent satellite trajectories inside the same 30 city boundaries
used by the Street View dataset. Satellite trajectories have their own IDs and
are not paired with Street View routes.

Each trajectory has **8 frames, 8 centers, and 7 movement transitions**.
Frame state is $S_i=(\mathbf c_i,\theta_i,e_i)$, for $i=0,\ldots,7$.
Each footprint is a rotated square with side length $e_i$ meters, where
$100\le e_i\le400$. Output images are 1024 x 1024 JPEGs. Heading is clockwise
from north; the final image's upward direction is the frame heading.

## 2. Sampling Distributions

All normal distributions below use **rejection sampling**: draw from the
specified underlying normal distribution, reject values outside the allowed
interval, and repeat. Do not clamp values to interval endpoints. The stated
mean and sigma describe the normal distribution before truncation; the observed
standard deviation after truncation is smaller. Geometry and imagery coverage
rejection can further affect the accepted dataset's empirical distributions.

| Quantity | Underlying distribution | Accepted interval |
| --- | --- | --- |
| Initial extent $e_0$ | Uniform | [100, 400] m |
| Initial heading $\theta_0$ | Uniform | [0, 360) degrees |
| Distance ratio $r_i=d_i/e_i$ | Normal(mean=0.4, sigma=0.2) | [0.2, 0.6] |
| Heading increment $\Delta\theta_i$ | Normal(mean=0, sigma=30 degrees) | [-30, 30] degrees |
| Extent increment $\Delta e_i$ | Normal(mean=0, sigma=50 m) | [-50, 50] m, additionally constrained by extent bounds |

### 2.1 Extent

For $i=1,\ldots,7$, sample

$$
\Delta e_i\sim\operatorname{TruncNormal}\left(0,50^2,
\max(-50,100-e_{i-1}),\min(50,400-e_{i-1})\right),
\qquad e_i=e_{i-1}+\Delta e_i.
$$

This preserves both $100\le e_i\le400$ and $|e_i-e_{i-1}|\le50$ m.
Near the extent limits, the allowed increment interval is asymmetric and the
conditional mean may differ from zero.

### 2.2 Heading

For $i=1,\ldots,6$:

$$
\Delta\theta_i\sim\operatorname{TruncNormal}(0,30^2,-30,30),
\qquad \theta_i=(\theta_{i-1}+\Delta\theta_i)\bmod360.
$$

The last frame has no outgoing transition, so $\theta_7=\theta_6$ and its
recorded turn is zero. The first frame's preceding turn is null. Store signed
turns in [-30, 30] degrees, using the shortest angular difference.

### 2.3 Center Movement

Set $\mathbf c_0=(0,0)$ in a local coordinate system with x east and y north.
For $i=0,\ldots,6$:

$$
r_i\sim\operatorname{TruncNormal}(0.4,0.2^2,0.2,0.6),
\qquad d_i=r_ie_i,
$$

$$
\mathbf c_{i+1}=\mathbf c_i+d_i
\begin{bmatrix}\sin\theta_i\\\cos\theta_i\end{bmatrix}.
$$

Distance is relative to the **current frame's** extent. For example, at
$e_i=300$ m the allowed distance is 60-180 m. The final frame's distance and
ratio are null. Each earlier heading equals the bearing to the next center.

## 3. Geographic Placement and Geometry Validation

1. Sample an anchor uniformly within the city boundary's longitude/latitude
   bounding box and reject anchors outside its polygons.
2. Sample a local trajectory. Reject center polylines with self-intersections.
3. Translate all local centers and square footprints to geographic coordinates
   using a local equirectangular approximation with Earth radius
   $R=6378137$ m and anchor latitude $\varphi_0$:
   $\varphi=\varphi_0+y/R$ and
   $\lambda=\lambda_0+x/(R\cos\varphi_0)$, with angles in radians.
4. Require the **entire polygon** of each rotated footprint to be covered by
   the city boundary union, including holes and concave boundaries. Checking
   centers or corners alone is insufficient.
5. Record adjacent footprint overlap as intersection area divided by the
   current frame footprint area. No minimum overlap or nonadjacent separation
   constraint is imposed.

Placement tries up to 4000 proposals. If imagery is unavailable after zoom
fallback, resample the placement and trajectory, up to 200 placements per ID.
Network failures retry each tile up to four times and then stop with an error;
completed trajectories can be resumed.

## 4. Dynamic Tile Zoom and Image Rendering

Source URL:

```text
https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}
```

### 4.1 Requested Zoom

For each frame independently, let $\varphi_i$ be its latitude, $e_i$ its extent,
$P=1024$ its output width, and $T=256$ the tile width. Web Mercator's approximate
ground resolution is

$$
\operatorname{mpp}(\varphi_i,z)=\frac{2\pi R\cos\varphi_i}{T2^z}.
$$

Select the lowest integer zoom whose extent contains at least P source pixels:

$$
z_i^{\mathrm{requested}}=
\operatorname{clip}_{[1,22]}\left(
\left\lceil\log_2\frac{2\pi R\cos\varphi_i\,P}{T e_i}\right\rceil
\right).
$$

Smaller extent requests higher zoom; latitude also affects the selected zoom.
Zoom is not fixed across a trajectory. Output ground resolution is always
$e_i/P$ m per output pixel. The maximum requested zoom defaults to 22.
The formula matches pixel sampling density; it does not guarantee the source
satellite sensor's native spatial resolution.

### 4.2 Coverage Fallback

If a requested mosaic contains HTTP 404 tiles, try the next lower zoom, down to
`min(18, requested_zoom)`. Record both requested and actual zoom. If no mosaic
is available at that limit, reject the placement. Lower zoom may require
upsampling; record this explicitly. Never replace missing imagery with blank
tiles for a valid city frame.

### 4.3 Rendering

1. Compute a source window of side
   `ceil(sqrt(2) * extent / source_mpp) + 8` pixels to cover rotation.
2. Fetch the required tile mosaic, with 16 concurrent downloads by default.
   Cache source tiles by zoom/x/y outside the generated trajectory directory.
3. Crop the source window around the frame center.
4. Rotate counterclockwise by the clockwise-from-north heading, using bicubic
   interpolation. This makes the selected world heading point upward.
5. Center-crop a square of `round(extent / source_mpp)` source pixels and resize
   to 1024 x 1024 with Lanczos. Save JPEG at quality 92.

Source crop rounding and the local map projection are approximations; output
extent and footprint labels remain the requested metric values.

## 5. Packaging and Reproducibility

Trajectory IDs follow `<city>__satellite_v4__<six-digit-index>`.
The ground truth is `gt_order=[0,1,2,3,4,5,6,7]`; stored images remain in this
order. Shuffling is a downstream training or evaluation step.

Each manifest stores the pipeline version, per-route seed, sampling parameters,
extent limits, image size, tile URL, dynamic zoom policy, validation flags,
and eight frame records. Frame records include:

```text
frame_id, file, center_x_m, center_y_m, latitude, longitude,
heading_deg, extent_m, distance_to_next_m, distance_extent_ratio,
turn_from_previous_deg, extent_change_from_previous_m, overlap_with_next,
requested_zoom, zoom, source_meters_per_pixel, output_meters_per_pixel,
source_crop_pixels, upsampled
```

For the last frame, outgoing distance, ratio and overlap are null; turn is zero.
For the first frame, preceding turn and extent change are null.

City indexes update atomically after each completed trajectory. Root and city
summaries report trajectory and image counts. Resume only complete manifests
matching this pipeline version; reuse cached tiles. Seed defaults to 20261001;
each route uses `seed + city_position * 100003 + route_index`. Coverage retries
consume the same random stream, so reproduction also depends on tile coverage.

## 6. The 30-Trajectory Example Run (Legacy v3)

Use the original 30 cities in `CITY_IDS` order. `--total-count 30` assigns one
trajectory to each city, producing 30 trajectories and 240 images. Remove the previous generated
trajectory dataset before this replacement run; retain source tile cache and
city boundaries. This run uses seed 20261001 and 32 tile workers. On the
production server, requests use the existing HTTP proxy at 127.0.0.1:18236;
the README contains the complete launch command.

The production viewer reads the replacement dataset at the existing
`datasets/satellite_trajectory_v1` directory. Its summary identifies this run as
`satellite_trajectory_v3_examples`. Selecting Satellite opens an independent
city/trajectory view with counts from the satellite indexes.
