const CITATION_RE = /\[S(\d+)\]/g;

export function parseCitations(text) {
  if (!text) return [];
  const parts = [];
  let lastIndex = 0;
  CITATION_RE.lastIndex = 0;
  let match;
  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: "citation", sid: `S${match[1]}` });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: "text", value: text.slice(lastIndex) });
  }
  return parts;
}

export function formatEvidenceLocation(location) {
  return location || "";
}
