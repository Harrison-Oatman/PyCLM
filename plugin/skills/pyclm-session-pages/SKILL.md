---
name: pyclm-session-pages
description: Make or update a session page - a static HTML page in the session folder that shows a design, the picked positions or a running experiment at a glance. Use before asking the person to approve a design or positions, when a run starts or is checked, or whenever a figure, image grid or plan would be clearer on a page than in the terminal.
---

# Session pages

A session page is something to look at beside the conversation. It is never
a way to control the microscope. The run goes on without it, and nothing on
the page sends a command. Approvals are always typed in the conversation.

Make a page when it helps the person see something. Three moments almost
always need one:

| Moment | What the page shows | When |
|---|---|---|
| **Design** | one tile per condition: the pattern on a sample image (`pyclm preview`), its label and dose; the light program as a time strip; the `pyclm check` result | before asking for approval of a design |
| **Positions** | the scout or stitched overview (`pyclm stitch`) with the picked fields outlined and numbered, colored by condition; one line on how they were picked | before asking for approval of positions |
| **Running** | status (timepoint of total, end time in local time, flags); the live analysis figure first; the newest frame of every position as a contact sheet | when a run starts, and at each check |

Build other pages whenever they help: a proposal for the next run with the
figure that motivates it, a comparison of two runs, a session summary. Use
the same style. Don't make a page for something one line in the terminal
says just as well.

The page adds to the conversation and never replaces it. Say the
decision and its context in the conversation too, so a person without a
browser loses nothing.

## Where pages live

- `<session folder>/pages/<moment>.html`, for example `pages/design_run1.html`,
  `pages/positions_run1.html`, `pages/running_run1.html`. Overwrite a page
  when it is updated, and keep one page per run and moment.
- Images go in `pages/img/`, referenced relatively, so the page opens from the file
  system (`file://`) with no server. Copy them there instead of linking into the
  run's data. Use PNG, at most about 1600 px wide.
- After writing a page, give the person its full path once (they open it
  in a browser, or in the desktop app's preview pane). After that, say
  "page updated".
- Never write pages inside an experiment directory that is running; `pyclm`
  owns that folder.

## Rules for what is on a page

1. **From files only.** Every number and image comes from a file you can
   name: `pyclm check` output, `status.json`, `frames.parquet`, `pyclm
   preview` or `stitch` images, a live-analysis figure. Put the source of each
   figure in the About panel.
2. **Figure first.** One title line with status pills, then the main
   figure or image grid. No paragraphs above it.
3. **Little text at rest.** Labels are short. Options, notes, sources and
   explanations go in panels that start closed, named from this fixed set:
   **Details**, **Sources**, **About**. Reuse these names on every page.
4. **Stale is visible.** A running page shows "updated HH:MM" and refreshes
   itself every 5 minutes (`<meta http-equiv="refresh" content="300">`). A pill
   turns to "stale" if the page is older than 45 minutes when it is
   rebuilt from a check.
5. **Flags never hide.** A flag is a pill in the title line and a mark on
   that position's tile; the run keeps going.
6. **No controls that act.** No buttons that write files, call commands or
   send messages. A "Figure view" toggle (white ground, no chrome) is allowed.
7. **Self-contained.** One HTML file plus `img/`. No external scripts.
   Fonts from Google Fonts with system fallbacks are fine; the page still
   works offline.

## Style

Use the CSS from `template.html` as it is: tokens, fonts and components.
The lab may override the tokens in `lab.md` ("Page style"); otherwise keep these.

- Paper ground `#f2ede3`, cards `#fbf9f4`, text `#33302a`, accent terracotta `#9a4f2c`.
- Data colors in this order, following the condition, never its rank:
  `#1e66f5 #e0661a #0a9199 #b98300 #d85fb0 #d20f39 #8839ef #40a02b`.
  A no-light control is gray `#a59d8e`.
- Light on: shading `rgba(70,175,215,0.27)`, edge `#38a9d1`. Light in
  diagrams is always light blue to cyan.
- Microscope images stay grayscale on black, or in their channel's
  usual color; never recolor them with data colors. Outline picked fields in
  their condition color.
- IBM Plex Sans for text, IBM Plex Serif for titles, IBM Plex Mono for numbers, paths
  and commands.
- matplotlib figures for a page: facecolor `#fbf9f4`, grid `#e3dccd`, text
  `#33302a`, the data colors above as the cycle, light-on as `axvspan`.

Components in the template: title line with pills (`ok`, `run`, `warn`,
`prop` for awaiting approval), tile grid (4 or 6 across, image + label with a
color swatch), figure block, card with a key/value list, legend, light-program
strip, closed panels. Copy the ones you need; don't invent new ones for
the three main pages.

## Checklist before telling the person

- Opens from the file system; images load; nothing scrolls sideways at 400 px.
- Every condition color matches the legend and the other pages of this
  session.
- The title line says what the page is about, which run, and its state.
- The conversation has the same decision in words.
