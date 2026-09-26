# PostgreSQL backup runbook

## Policy

- Create one PostgreSQL custom-format dump every day.
- Store dumps under `data/backups/` with mode `600`.
- Keep 30 days by default.
- Verify every new archive with `pg_restore --list` before publishing it.
- Treat every dump as private transcript data.

## Install

```bash
chmod +x scripts/backup_postgres.sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/plaud-brain-backup.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now plaud-brain-backup.timer
```

## Run and inspect

```bash
systemctl --user start plaud-brain-backup.service
systemctl --user status plaud-brain-backup.service
systemctl --user list-timers plaud-brain-backup.timer
journalctl --user -u plaud-brain-backup.service -n 50 --no-pager
find data/backups -maxdepth 1 -type f -name 'plaud_brain-*.dump' -ls
```

Override `BACKUP_DIR` or `BACKUP_RETENTION_DAYS` in the service only after
validating the target directory. Retention must be between 1 and 3650 days.

## Restore test

Never test a restore against the production database. Create a disposable
database, feed one dump to `pg_restore`, validate representative counts, and
drop only that explicitly named test database.

```bash
docker compose exec -T postgres createdb -U plaud plaud_brain_restore_test
docker compose exec -T postgres pg_restore -U plaud -d plaud_brain_restore_test \
  < data/backups/<backup-file>.dump
docker compose exec -T postgres psql -U plaud -d plaud_brain_restore_test \
  -c 'select count(*) from recordings;'
docker compose exec -T postgres dropdb -U plaud plaud_brain_restore_test
```

## Limits

The timer backs up PostgreSQL only. Separately protect `data/audio/`, the
encrypted `.env`, and `~/.plaud/`. Do not place unencrypted dumps in Git or
cloud-synchronized public folders.
