import { parseCitations } from "../lib/citations";
import CitationChip from "./CitationChip";
import LoadingIndicator from "./LoadingIndicator";

function formatDuration(durationMs) {
  if (durationMs == null) return null;
  return `${(durationMs / 1000).toFixed(1)}sn`;
}

export default function AnswerCard({ turn, selectedSid, onSelectEvidence }) {
  const { question, response, error, pending, durationMs } = turn;

  if (pending) {
    return (
      <div id={`turn-${turn.id}`} className="flex flex-col gap-2.5 animate-bubble-morph">
        <div className="flex justify-end">
          <p className="bg-surface-2 rounded-[14px_14px_4px_14px] px-3.5 py-2.5 max-w-[78%] text-[13.5px] font-semibold m-0">
            {question}
          </p>
        </div>
        <div className="flex justify-start">
          <LoadingIndicator />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div id={`turn-${turn.id}`} className="flex flex-col gap-2.5 animate-bubble-morph">
        <div className="flex justify-end">
          <p className="bg-surface-2 rounded-[14px_14px_4px_14px] px-3.5 py-2.5 max-w-[78%] text-[13.5px] font-semibold m-0">
            {question}
          </p>
        </div>
        <p className="text-risk-high">{error}</p>
      </div>
    );
  }

  const { answer, evidence, answer_error } = response;
  const evidenceBySid = new Map((evidence || []).map((e) => [e.sid, e]));
  const duration = formatDuration(durationMs);

  return (
    <div id={`turn-${turn.id}`} className="flex flex-col gap-2.5 animate-bubble-morph">
      <div className="flex justify-end">
        <p className="bg-surface-2 rounded-[14px_14px_4px_14px] px-3.5 py-2.5 max-w-[78%] text-[13.5px] font-semibold m-0">
          {question}
        </p>
      </div>
      <div className="flex justify-start">
        {answer ? (
          <div className="max-w-[88%]">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="font-mono font-bold text-[10px] tracking-wide uppercase text-text bg-surface-2 border border-accent rounded px-1.5 py-0.5">
                ÖZ
              </span>
              {duration && <span className="font-mono text-[10.5px] text-text-muted">{duration}</span>}
            </div>
            <p className="m-0 leading-[1.65] text-[13.5px]">
              {parseCitations(answer.answer_text).map((part, i) =>
                part.type === "citation" ? (
                  <CitationChip
                    key={i}
                    sid={part.sid}
                    active={selectedSid === part.sid}
                    onClick={() => {
                      const e = evidenceBySid.get(part.sid);
                      if (e) onSelectEvidence(e);
                    }}
                  />
                ) : (
                  <span key={i}>{part.value}</span>
                )
              )}
            </p>
          </div>
        ) : (
          <p className="text-text-muted italic">{answer_error || "(cevap yok)"}</p>
        )}
      </div>
    </div>
  );
}
