#!/usr/bin/env python3
"""hst_collapse_headers.py FILE...  -- collapse the repeated headers a restart leaves in a .hst

AthenaK re-emits the two-line history header every time the run starts, so a series that
survived many relaunches carries one header pair per launch, most of them with no rows behind
them.  Readers that skip a fixed number of lines then mis-parse the file.  This keeps the first
header, drops every later one, and does not touch a single data row.  Rows are left exactly as
written: if two launches really did cover the same time, that is a data question, not a
formatting one, and it is reported rather than resolved here."""
import sys, shutil

for path in sys.argv[1:]:
    lines = open(path).readlines()
    out, seen_header, headers, rows = [], False, 0, []
    i = 0
    while i < len(lines):
        if lines[i].startswith('#') and 'history data' in lines[i]:
            headers += 1
            block = lines[i:i + 2] if i + 1 < len(lines) and lines[i + 1].startswith('#') else lines[i:i + 1]
            if not seen_header:
                out.extend(block); seen_header = True
            i += len(block); continue
        if not lines[i].startswith('#'):
            f = lines[i].split()
            if f:
                try: rows.append(float(f[0]))
                except ValueError: pass
        out.append(lines[i]); i += 1

    back = sum(1 for k in range(1, len(rows)) if rows[k] <= rows[k - 1])
    shutil.copy2(path, path + '.orig')
    open(path, 'w').writelines(out)
    print(f"{path}: {headers} headers -> 1, {len(rows)} data rows kept "
          f"({len(lines)} -> {len(out)} lines); non-increasing time steps: {back}")
    if back:
        print("   ^ overlapping launches present -- decide which branch to keep before plotting")
