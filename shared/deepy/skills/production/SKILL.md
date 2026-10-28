---
name: production
description: Plan, make, check and report a production that combines several generated media (a film or episode, an ad or campaign, an album, a game or asset pack, a series), when the request only describes the result.
---

# Production

Users describe what they want; the method is yours. Apply it without being asked.

## 1. Plan first

- Before generating, write a short plan in `@documents`: the deliverables, the facts every asset must agree on (names, numbers, dates, looks, lyrics, rules), the shot list or track list with lengths, what is made once and reused, the models chosen and the expected GPU time.
- Keep later steps consistent with the plan, and update it when something changes.

## 2. Make

- Use the configured templates and their fast or accelerated variants; keep video at the template's resolution (720p or below) unless the user asks otherwise.
- Make shared assets once (master images of recurring people and objects, voices, music) and reuse them; submit independent generations together in one batch.
- Match the pace to the genre: ads, trailers and music videos change shot every few seconds; a shot held for long must be worth watching that long.
- Every video gets sound under its whole length, music or narration over ambience, unless the user asks for silence: a measured silent stretch is a defect to fix, not a question for the user.
- Draw designed on-screen text (titles, lower thirds, labels, signs, UI screens, timers, scores) over the finished footage with `draw_svg` `video_id` rather than asking a video model to write it: the drawing is exact and the model never redraws it. Subtitles are never burned in: they go in a subtitle track that viewers can switch off.
- Give every recurring person and object its master image as start image or reference in each shot that shows it. Make start images in the delivery's aspect ratio: video models keep the shape of their start image. For films, follow wangp://skills/edited-film; for a song-driven video, wangp://skills/music-video.

## 3. Check before answering

- Check every deliverable against the request and the plan: the content asked for, people and objects identical from the first to the last shot, every visible text and number against the documents, lengths and order, sound (speech complete and in sync, music under the whole cut, no black or silent stretch).
- Ask inspectors to describe, not to confirm. Measured values (`black_ranges_seconds`, `cut_times_seconds`, `quiet_ranges_seconds`, `voice_pitch_hz`, transcriptions) override impressions. Fix what fails rather than explaining it away.
- Judge a generated shot over its whole length: a few sampled frames can pass while the shot cuts away between them.
- A video model redraws the text of its start image: read every on-screen text and number in the finished video itself, at its start, middle and end, from zoomed crops, character by character. A cut inside one shot (`cut_times_seconds` of the generation result) is the model switching to another shot: keep the part before it or regenerate.
- When a shot fails twice for the same reason, change the approach instead of rerolling: a shorter shot (long shots drift and cut away more), another template, a cut around the defect, other footage over it, or report the limitation. Note each attempt with its media ID and verdict in the plan, so it survives a context compaction.

## 4. Report

- List the deliverables with media IDs and paths, how each was made, what was checked with the measured results, and any limitation that remains. Call "requested" only what the user asked for: a template default or your own choice is not a request.
