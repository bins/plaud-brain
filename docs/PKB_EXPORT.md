# PKB export

`plaud-brain export-pkb` projects recordings into the PKB knowledge base
(`bins/pkb`) as Markdown files. PKB treats Markdown as canonical and indexes
every change it sees, with local `bge-m3` embeddings.

## Ownership

The export folder is a **projection owned by plaud-brain**. PostgreSQL here
remains the source of truth; every file is regenerated from it. Do not edit
exported files by hand: the next export overwrites them. Put your own notes
elsewhere in the PKB root.

## Setup on bmax-ai

```bash
# A folder inside the PKB root, owned by the user that runs plaud-brain.
install -d -m 0755 /home/bins/knowledge/imports/plaud
```

```dotenv
PKB_EXPORT_DIR=/home/bins/knowledge/imports/plaud
PKB_EXPORT_AUTO=true
PKB_EXPORT_CLASSIFICATION=private   # or sensitive
PKB_EXPORT_FILE_MODE=644
```

PKB's containers run as UID 65532 and mount the root read-only, so they must
be able to traverse the parent folders and read the files. Mode `644` is the
simple choice when the knowledge root itself is private (for example `0711` on
the root owned by you). To keep files unreadable by other local users, use
`PKB_EXPORT_FILE_MODE=640` and give UID 65532 read access with an ACL:

```bash
setfacl -m u:65532:rx /home/bins/knowledge/imports/plaud
setfacl -d -m u:65532:r /home/bins/knowledge/imports/plaud
```

Do not point `PKB_EXPORT_DIR` at PKB's `inbox/`: that namespace belongs to the
capture API and its file quota.

Run once by hand and check that PKB indexed the files:

```bash
.venv/bin/plaud-brain export-pkb
# {"exported": N, "export_unchanged": 0, "export_deleted": 0, "export_failed": 0}
```

## File format

One file per recording: `<plaud_id>.md`.

- Front matter: `title`, `type: plaud-recording`, `created` (recording time),
  `updated` (summary time), `tags: [plaud, voice]`.
- `security`: the configured classification, `cloud_ai: deny`,
  `local_ai: allow`. Transcripts contain other people's speech, so cloud access
  is never granted by the exporter.
- `source` is a mapping (`type: plaud`, `plaud_id`), as PKB requires.
- Everything else lives under `metadata`, because PKB drops unknown top-level
  keys: `generator: plaud-brain-export`, `schema_version`, `recording_id`, STT
  and summary provider/model.
- Body: `# <name>`, then `## Summary`, `## Topics`, `## Decisions`,
  `## Action items` (with `[x]` for done and `(dismissed)`), and
  `## Transcript` with a `### HH:MM:00` heading every 5 minutes. One transcript
  segment per line, so PKB's line-range citations map to audio timestamps.

## Safety rules

- **Deterministic output.** No export timestamp is written, so an unchanged
  recording keeps identical bytes and PKB does not re-index it.
- **Atomic writes.** Each file is written to a hidden `.tmp` file in the same
  folder, flushed, then renamed. PKB's watcher ignores hidden and non-Markdown
  files.
- **Safe names.** Filenames come only from `plaud_id` matching
  `[A-Za-z0-9_-]+`; recording names never become paths.
- **No injection.** Front-matter values are written as escaped double-quoted
  YAML strings, and model or transcript text cannot open headings or code
  fences in the body.
- **Careful deletion.** A file is deleted only when its recording is no longer
  exportable, the file carries the `generator: plaud-brain-export` marker, the
  whole export pass succeeded, and at least one recording was found (so a wrong
  `DATABASE_URL` cannot empty the folder).

Verified by parsing exported files with PKB's own parser
(`internal/ingest.ParseFile`): policy, `source`, and `metadata` round-trip,
and the transcript splits into one chunk per 5-minute window.
