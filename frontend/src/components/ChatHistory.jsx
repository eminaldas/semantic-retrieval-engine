import AnswerCard from "./AnswerCard";

export default function ChatHistory({ turns, selectedSid, onSelectEvidence, onForceAnswer }) {
  if (turns.length === 0) {
    return <p className="text-text-muted max-w-[700px] w-full mx-auto px-6 box-border">Henüz bir soru sormadınız.</p>;
  }
  return (
    <div className="flex-1 flex flex-col gap-4 max-w-[700px] w-full mx-auto px-6 box-border">
      {turns.map((turn) => (
        <AnswerCard key={turn.id} turn={turn} selectedSid={selectedSid} onSelectEvidence={onSelectEvidence} onForceAnswer={onForceAnswer} />
      ))}
    </div>
  );
}
