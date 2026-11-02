"""Small Markdown-table formatter without optional dataframe dependencies."""
import numbers
import pandas as pd


def markdown_table(frame, index=True):
    table = frame.reset_index() if index else frame
    def format_cell(value):
        if isinstance(value, tuple):
            return " / ".join(str(v) for v in value if str(v))
        if isinstance(value, numbers.Real) and not isinstance(value, (int, bool)):
            return f"{value:.9g}" if pd.notna(value) else "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(format_cell(c) for c in table.columns) + " |",
            "| " + " | ".join("---" for c in table.columns) + " |"]
    rows.extend("| " + " | ".join(format_cell(v) for v in row) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(rows)
