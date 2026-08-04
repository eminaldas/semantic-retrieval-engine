export default function HistoryPanel({ turns, onSelectTurn }) {
  if (turns.length === 0) {
    return <p className="text-text-muted text-[13px]">Henüz bir soru sormadınız.</p>;
  }
  return (
    <ul className="list-none m-0 p-0 flex flex-col gap-2">
      {turns.map((t) => (
        <li
          key={t.id}
          onClick={() => onSelectTurn(t.id)}
          className="px-2 py-1.5 rounded-[10px] bg-surface border border-border text-[13px] cursor-pointer transition-colors duration-150 hover:border-border-hover"
        >
          {t.question}
        </li>
      ))}
    </ul>
  );
}
