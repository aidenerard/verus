# Canonical Rebar Depth Map — locked output spec

The rebar depth map is a **fixed, dataset-independent** output. Every analysis —
DZT (Infrasense) or Proceq (Terracon) — must render to the exact same format so
`color == depth` is identical across all jobs. Reference fixture:
[`depth_map_reference.png`](depth_map_reference.png).

## The single source of truth

`server/analysis.py :: build_unified_depth_map()` is the ONLY depth-map renderer.
Both pipelines call it (Proceq directly with GPS coords; DZT via
`pipeline._render_depth_b64_via_unified` with the 2D depth grid). The frontend
**displays the PNG it produces** (`result.rebar_depth_map_url` /
`rebar_depth_image`) — it must never re-render the depth map client-side.

## Locked parameters — do not change without updating the fixture

| Property | Value |
|---|---|
| Colormap | `YlOrRd_r` (dark red = shallow, pale yellow = deep) |
| Color range | **vmin = 3.0″, vmax = 8.5″** (FIXED — never auto-scale to a dataset) |
| Color bands | discrete 0.5″ steps (`levels = arange(3.0, 9.0, 0.5)`) |
| Depth grouping | values snapped to the nearest whole inch (`np.round`) |
| Isolines | black, on **integer inches only**, each labelled with its inch value |
| Colorbar | horizontal, bottom, ticks at every 0.5″ level, `extend='neither'` |
| Background | white; title = `analysis_name` verbatim |
| Axes | bare integer ticks when no dimensions given. When the inspector enters **bridge length × width (ft)** (`bridge_length_ft`/`bridge_width_ft`), the plan-view is linearly rescaled to real feet: labelled "Distance Along Bridge (ft)" × "Bridge Width (ft)", adaptive ~10–14 ticks, and the title appends " — L ft × W ft". Palette/range/banding are unchanged. |

## Rules

- **Never** reintroduce a per-dataset / auto-scaling depth renderer (the old
  `DepthMapCanvas` viridis canvas was removed for exactly this reason — it
  rescaled colors per job, so the map "kept changing").
- If the format must change, regenerate `depth_map_reference.png` in the same
  commit and update this table.
- The corrosion map is likewise unified via `render_corrosion_db_map()`
  (RdYlGn, fixed −20→0 dB, −8 dB ASTM D6087 at-risk line).
