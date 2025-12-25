#!/usr/bin/env python3
"""
Convert a CSV with columns like:
  First Name, Last Name, SID, Email, Sections,
  Quiz 1, Quiz 1 - Max Points, Quiz 1 - Submission Time, Quiz 1 - Lateness (H:M:S),
  Pset 2, Pset 2 - Max Points, ...
into a Canvas Gradebook-importable CSV.

Canvas CSV shape:
  Columns: Student, ID, SIS User ID, SIS Login ID, Section, <Assignment 1>, <Assignment 2>, ...
  Row 2:   "Points Possible" in Student column, with max points in each assignment column.

Usage:
  python to_canvas.py input.csv output.csv
"""

import sys
import pandas as pd
from pathlib import Path

# ---- Configuration you can tweak if needed ----
ROSTER_MAP = {
    "Student": ("Last Name", "First Name"),  # will be formatted as "Last, First"
    "SIS User ID": "SID",
    "SIS Login ID": "Email",
    "Section": "Sections",
}
# Columns that are definitely NOT assignment score columns
ALWAYS_IGNORE = {
    "First Name",
    "Last Name",
    "SID",
    "Email",
    "Sections",
    "Total Lateness (H:M:S)",
}

def guess_columns(df):
    """
    Identify:
      - score_columns: the base assignment columns like 'Quiz 1', 'Pset 2', 'Pset 2 Self-grade', 'Extra Credit'
      - max_columns: mapping from base assignment -> matching '<base> - Max Points'
    We ignore submission time/lateness columns.
    """
    cols = list(df.columns)

    # Build a mapping from "base name" -> "Max Points" col
    max_for = {}
    for c in cols:
        if c.endswith(" - Max Points"):
            base = c.replace(" - Max Points", "").strip()
            max_for[base] = c

    # Candidate score columns are those that:
    #  - are not in ALWAYS_IGNORE
    #  - do NOT contain " - " (so we avoid time/lateness/max)
    #  - appear in the CSV (obviously)
    # We’ll also keep things like "Pset 3 Self-grade" as their own assignments.
    score_columns = []
    for c in cols:
        if c in ALWAYS_IGNORE:
            continue
        if " - " in c:
            continue
        # Heuristic: keep numeric-ish columns (scores often are numeric or blank)
        # but don't over-constrain—Canvas will accept blanks too.
        score_columns.append(c)

    # Preserve original order as they appeared
    return score_columns, max_for


def build_canvas_dataframe(df):
    score_columns, max_for = guess_columns(df)

    # Build the roster columns for Canvas
    # Canvas accepts either ID (Canvas User ID), SIS User ID, or SIS Login ID to match users.
    # We’ll include SIS User ID (your 'SID') and SIS Login ID (email) for robust matching.
    out_cols = ["Student", "ID", "SIS User ID", "SIS Login ID", "Section"]
    # Add assignments after roster columns
    out_cols.extend(score_columns)

    # Prepare the main data rows
    out_rows = []

    for _, row in df.iterrows():
        # Compose Student "Last, First"
        last = row.get("Last Name", "")
        first = row.get("First Name", "")
        student_name = f"{str(last).strip()}, {str(first).strip()}".strip(", ")

        out = {
            "Student": student_name,
            "ID": "",  # leave blank; Canvas will match on SIS User ID or SIS Login ID
            "SIS User ID": row.get("SID", ""),
            "SIS Login ID": row.get("Email", ""),
            "Section": row.get("Sections", ""),
        }

        # Copy scores
        for a in score_columns:
            val = row.get(a, "")
            # Try to coerce to float (Canvas likes numbers); keep blank if non-numeric
            try:
                if pd.isna(val) or str(val).strip() == "":
                    out[a] = ""
                else:
                    out[a] = float(val)
            except Exception:
                out[a] = "" if pd.isna(val) else val
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows, columns=out_cols)

    # Insert the "Points Possible" row as the SECOND row
    # Canvas expects: "Student" column contains "Points Possible",
    # roster identifiers blank in that row, assignments filled with max points.
    pp_row = {c: "" for c in out_cols}
    pp_row["Student"] = "Points Possible"

    for a in score_columns:
        max_col = max_for.get(a)
        if max_col is None:
            # If there's no explicit max column, leave blank (Canvas will keep existing max or you can set it later)
            pp_row[a] = ""
        else:
            # Some files repeat max points per student; take the first non-null max found
            # across the file for consistency.
            max_val = pd.to_numeric(df[max_col], errors="coerce").dropna()
            if len(max_val) > 0:
                pp_row[a] = float(max_val.iloc[0])
            else:
                pp_row[a] = ""

    # Place the Points Possible row after the header (index 0)
    out_df = pd.concat([pd.DataFrame([pp_row], columns=out_cols), out_df], ignore_index=True)

    return out_df


def main(in_path, out_path):
    df = pd.read_csv(in_path, sep=None, engine="python").fillna("")
    # Normalize header names a bit (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    out_df = build_canvas_dataframe(df)

    # Write CSV exactly as Canvas expects (comma-separated, UTF-8, no index)
    out_df.to_csv(out_path, index=False)
    print(f"✔ Wrote Canvas import CSV to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} input.csv output_canvas.csv")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
