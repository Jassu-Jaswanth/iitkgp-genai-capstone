import { useState } from "react";

/** Pretty JSON, folded when long. */
export default function Json({ value, maxLines = 30 }: { value: unknown; maxLines?: number }) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  const lines = (text ?? "").split("\n");
  const [open, setOpen] = useState(lines.length <= maxLines);
  if (value === null || value === undefined || text === "") return <span className="muted">(empty)</span>;
  return (
    <div className="json">
      <pre>{open ? text : lines.slice(0, maxLines).join("\n") + "\n…"}</pre>
      {lines.length > maxLines && (
        <button className="link" onClick={() => setOpen(!open)}>
          {open ? "Show less" : `Show all ${lines.length} lines`}
        </button>
      )}
    </div>
  );
}
