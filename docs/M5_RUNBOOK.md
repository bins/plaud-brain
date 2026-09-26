# M5 local web runbook

## Start and inspect

```bash
systemctl --user enable --now plaud-brain-web.service
systemctl --user status plaud-brain-web.service
ss -ltn | grep 8787
```

The expected listener is `127.0.0.1:8787`, never `0.0.0.0:8787`.

## Connect

From the client machine:

```bash
ssh -L 8787:127.0.0.1:8787 bins@bmax-ai
```

Open `http://127.0.0.1:8787`. Keep the terminal session open while using the
interface.

Keep the repository `.env` file at mode `600` because it contains provider
and database credentials.

## Verify

```bash
curl -fsS http://127.0.0.1:8787/api/health
curl -I http://127.0.0.1:8787/
journalctl --user -u plaud-brain-web.service -n 50 --no-pager
```

Health should return `{"status":"ok"}`; the page should include CSP,
`X-Frame-Options: DENY`, and `X-Content-Type-Options: nosniff`.

## Recovery

```bash
systemctl --user restart plaud-brain-web.service
systemctl --user stop plaud-brain-web.service
```

Stopping the web service does not affect scheduled ingestion or PostgreSQL.
