export default function CitationChip({ sid, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        "inline-flex items-center justify-center min-w-[22px] h-5 px-1 mx-0.5 font-mono text-[11px] rounded border cursor-pointer transition-colors duration-150 " +
        (active ? "border-accent text-accent-strong" : "border-border-hover text-text hover:border-accent hover:text-accent-strong")
      }
    >
      {sid}
    </button>
  );
}
