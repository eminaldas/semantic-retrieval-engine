import Icon from "./Icon";

export default function SourceRow({ source, job, confirming, onDeleteClick, onCancelDelete, onConfirmDelete, onPreview }) {
  if (job && job.status === "running") {
    return (
      <li className="flex flex-col items-stretch gap-1 p-2 rounded-md text-[13px] bg-surface border border-border">
        <span className="font-semibold break-all">{source.source_id}</span>
        <span className="text-accent-strong text-xs">
          İşleniyor: {job.step_name} ({job.step}/{job.total_steps})
        </span>
      </li>
    );
  }

  if (job && job.status === "error") {
    return (
      <li className="flex flex-col items-stretch gap-1 p-2 rounded-md text-[13px] bg-surface border border-border">
        <span className="font-semibold break-all">{source.source_id}</span>
        <span className="text-risk-high text-xs">Hata: {job.error_message} (tekrar deneyin)</span>
      </li>
    );
  }

  if (confirming) {
    return (
      <li className="flex flex-col items-stretch gap-1 p-2 rounded-md text-[13px] bg-surface border border-border">
        <span className="font-semibold break-all">{source.source_id}</span>
        <span className="flex items-center gap-1.5 text-xs">
          Emin misiniz?
          <button
            type="button"
            onClick={onConfirmDelete}
            className="border border-border bg-surface-2 text-text rounded-md px-2 py-0.5 cursor-pointer"
          >
            Evet
          </button>
          <button
            type="button"
            onClick={onCancelDelete}
            className="border border-border bg-surface-2 text-text rounded-md px-2 py-0.5 cursor-pointer"
          >
            İptal
          </button>
        </span>
      </li>
    );
  }

  return (
    <li
      onClick={() => onPreview(source.source_id)}
      className="group relative flex items-center gap-2 p-2 rounded-md text-[13px] cursor-pointer transition-colors duration-150 hover:bg-surface-2"
    >
      <Icon
        name="document"
        className="w-[15px] h-[15px] text-text-muted shrink-0 transition-colors duration-150 group-hover:text-accent"
      />
      <span className="font-semibold break-all flex-1 min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
        {source.source_id}
      </span>
      <div className="flex gap-0.5 shrink-0 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
        <button
          type="button"
          title="Güncelle"
          onClick={(ev) => ev.stopPropagation()}
          aria-label="Güncelle"
          className="w-5 h-5 border-0 bg-transparent text-text-muted rounded cursor-pointer flex items-center justify-center transition-colors duration-150 hover:text-accent-strong hover:bg-white/[0.06] [&_svg]:w-3 [&_svg]:h-3"
        >
          <Icon name="edit" />
        </button>
        <button
          type="button"
          title="Sil"
          onClick={(ev) => {
            ev.stopPropagation();
            onDeleteClick();
          }}
          aria-label="Sil"
          className="w-5 h-5 border-0 bg-transparent text-text-muted rounded cursor-pointer flex items-center justify-center transition-colors duration-150 hover:text-accent-strong hover:bg-white/[0.06] [&_svg]:w-3 [&_svg]:h-3"
        >
          <Icon name="trash" />
        </button>
      </div>
      <div className="absolute left-2 right-2 top-full mt-0.5 z-[5] bg-surface-2 border border-border-hover rounded-md px-2.5 py-1.5 font-mono text-[10.5px] text-text-muted shadow-strong opacity-0 pointer-events-none transition-opacity duration-150 group-hover:opacity-100">
        {source.chunk_count} parça · {source.entity_count} varlık · {source.relation_count} ilişki
      </div>
    </li>
  );
}
