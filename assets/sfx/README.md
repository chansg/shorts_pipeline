# SFX library

Timelines reference sound cues by **tag**, never by path. This file maps tags to
files so a manifest stays portable.

```
assets/sfx/
├─ sfx_map.json      # { "<tag>": "<relative path under assets/sfx/>" }
├─ library/          # bundled cues (license-free, synthesized with ffmpeg)
└─ imported/         # your loudnorm'd mp3 imports land here as @import/<name>
```

## Tags

| tag | what it is | typical layer |
|-----|------------|---------------|
| `knock_wood`  | two short wooden knocks | one-shot |
| `wind_hall`   | low filtered wind, loopable | ambient_bed |
| `rot_shimmer` | high shimmering tremolo | motif |
| `boom_low`    | deep low impact | one-shot |

The four bundled cues are **synthesized** (ffmpeg `lavfi`), so they are tiny and
carry no licensing restrictions. Drop your own `.wav`/`.mp3` into `library/` and
add a line to `sfx_map.json` to grow the set, or use the mp3 import flow
(`--sfx-import`) which registers an `@import/<name>` tag automatically.

## Referencing a cue

In a manifest, a cue's `source` is one of:

- a **tag** from this map — `"source": "knock_wood"`
- an **import** alias — `"source": "@import/my_track"`
- a raw **path** (absolute, or relative to the repo root) — escape hatch only
