import { formatEvidenceLocation } from "../lib/citations";
import Icon from "./Icon";

export default function EvidenceDrawerContent({ evidence, sourceContent }) {
  if (!evidence && !sourceContent) {
    return <p className="text-text-muted text-[13px]">Bir atıfa ya da kaynağa tıklayın.</p>;
  }

  const sourceId = evidence ? evidence.source_id : sourceContent.source_id;

  return (
    <div className="flex flex-col gap-4 text-[13px]">
      <div className="bg-surface-2 border border-border rounded-lg px-3 py-2.5">
        <div className="flex items-center gap-2 pb-2 mb-2 border-b border-border">
          <div className="w-5 h-5 rounded-md bg-surface text-text-muted flex items-center justify-center shrink-0 [&_svg]:w-3 [&_svg]:h-3">
            <Icon name="document" />
          </div>
          <span className="font-bold text-xs break-all">{sourceId}</span>
        </div>
        {evidence && (
          <p className="text-text-muted font-mono text-[10.5px] m-0">
            Konum: {formatEvidenceLocation(evidence.location)}
          </p>
        )}
      </div>
      {evidence && (
        <p className="m-0 italic text-text-muted border-l-2 border-accent pl-2.5 leading-[1.55]">
          &ldquo;{evidence.quote}&rdquo;
        </p>
      )}
      {sourceContent && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-text-muted mb-2">Kaynağın tamamı</p>
          <div className="bg-bg border border-border rounded-lg px-3 py-2.5">
            {sourceContent.chunks.map((c) => (
              <p
                key={c.chunk_id}
                className={
                  "font-mono text-[11px] leading-[1.7] mb-1 " +
                  (evidence && evidence.chunk_id === c.chunk_id
                    ? "text-accent-strong border-l-2 border-accent pl-2 -ml-px"
                    : "text-text-muted")
                }
              >
                {formatEvidenceLocation(c.location)} — {c.text}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
