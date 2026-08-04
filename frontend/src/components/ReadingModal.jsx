import { useEffect } from "react";
import { formatEvidenceLocation } from "../lib/citations";
import Icon from "./Icon";

export default function ReadingModal({ open, onClose, evidence, sourceContent }) {
  useEffect(() => {
    if (!open) return undefined;
    function onKeyDown(ev) {
      if (ev.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open || (!evidence && !sourceContent)) return null;

  const sourceId = evidence ? evidence.source_id : sourceContent.source_id;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 bg-black/55 flex items-center justify-center z-40 animate-modal-fade"
    >
      <div
        onClick={(ev) => ev.stopPropagation()}
        className="w-[82%] h-[84%] max-w-[920px] bg-surface border border-border-hover rounded-xl shadow-strong flex flex-col overflow-hidden animate-modal-scale-in"
      >
        <div className="flex items-center justify-between px-[18px] py-3 border-b border-border shrink-0">
          <span className="font-mono text-[13px] font-bold break-all">{sourceId}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Kapat"
            className="border border-border bg-surface-2 text-text-muted rounded-md w-[26px] h-[26px] flex items-center justify-center cursor-pointer shrink-0 transition-colors duration-150 hover:border-accent hover:text-text [&_svg]:w-[13px] [&_svg]:h-[13px]"
          >
            <Icon name="close" />
          </button>
        </div>
        <div className="reading-modal-body flex-1 overflow-y-auto py-6 px-8 max-w-[680px] mx-auto w-full box-border">
          {sourceContent ? (
            sourceContent.chunks.map((c) => (
              <p
                key={c.chunk_id}
                className={
                  "font-mono text-[13px] leading-[1.85] mb-3.5 " +
                  (evidence && evidence.chunk_id === c.chunk_id
                    ? "text-accent-strong border-l-2 border-accent pl-2.5 -ml-0.5"
                    : "text-text-muted")
                }
              >
                {formatEvidenceLocation(c.location)} — {c.text}
              </p>
            ))
          ) : (
            <p className="font-mono text-[13px] leading-[1.85] mb-3.5 text-accent-strong border-l-2 border-accent pl-2.5 -ml-0.5">
              {formatEvidenceLocation(evidence.location)} — {evidence.quote}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
