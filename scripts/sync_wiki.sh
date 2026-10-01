#!/usr/bin/env bash
# Publish the wiki pages in docs/ to the GitHub wiki of this repository.
#
# Usage:  scripts/sync_wiki.sh [--dry-run] [WIKI_GIT_URL]
#   WIKI_GIT_URL defaults to https://github.com/HongxuanJiang/athenak-transients.wiki.git
#   --dry-run    do everything except the push, and leave the clone for inspection
#
# Page names: docs/wiki/README.md -> Home, docs/eos_tables.md -> EOS-Tables,
# docs/remap_usage.md -> Remap-Usage, docs/wiki/<Name>.md -> <Name>.  The sidebar is
# generated.  Pages in the wiki that are not in this list are left alone.
# The wiki repository must exist: create any page once in the GitHub web UI first.
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
URL=${1:-https://github.com/HongxuanJiang/athenak-transients.wiki.git}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(mktemp -d)
trap '[ "$DRY" -eq 1 ] || rm -rf "$WORK"' EXIT

git clone --quiet "$URL" "$WORK/wiki"
cd "$WORK/wiki"

cp "$ROOT/docs/wiki/README.md" Home.md
cp "$ROOT/docs/eos_tables.md" EOS-Tables.md
cp "$ROOT/docs/remap_usage.md" Remap-Usage.md
for f in "$ROOT"/docs/wiki/*.md; do
  name=$(basename "$f" .md)
  [ "$name" = "README" ] && continue
  cp "$f" "$name.md"
done

cat > _Sidebar.md <<'SIDE'
**[Home](Home)**

**Local adaptive time stepping**
- [Overview](Local-Adaptive-Time-Stepping)
- [Implementation notes](Local-Adaptive-Time-Stepping-Implementation-Notes)

**Restart remap**
- [Remap usage](Remap-Usage)
- [Remapping](Remapping)
- [Implementation notes](Remapping-Implementation-Notes)

**Tabulated EOS**
- [Tabulated EOS](Tabulated-EOS)
- [EOS tables](EOS-Tables)
- [Implementation notes](Tabulated-EOS-Implementation-Notes)

**Multigrid self-gravity**
- [Overview](Multigrid-Self-Gravity)
- [Implementation notes](Multigrid-Self-Gravity-Implementation-Notes)

**Dual energy**
- [Overview](Dual-Energy)
- [Implementation notes](Dual-Energy-Implementation-Notes)
SIDE

git add -A
if git diff --cached --quiet; then echo "wiki is already up to date"; exit 0; fi
git diff --cached --stat | tail -20
git commit --quiet -m "Sync wiki from docs/ of the main repository"
if [ "$DRY" -eq 1 ]; then echo "dry run: not pushed; clone is $WORK/wiki"; exit 0; fi
git push
echo "wiki updated"
