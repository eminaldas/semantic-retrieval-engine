import Icon from "./Icon";

const ICON_BTN =
  "border border-border bg-surface-2 text-text-muted rounded-md w-6 h-6 flex items-center justify-center cursor-pointer transition-colors duration-150 hover:border-accent hover:text-text [&_svg]:w-3 [&_svg]:h-3";

export default function Drawer({ open, onOpen, onClose, onMaximize, children }) {
  return (
    <div className="flex h-full">
      <button
        type="button"
        onClick={onOpen}
        hidden={open}
        aria-label="Kanıt panelini aç"
        className="group relative w-8 border-0 border-l border-border bg-surface cursor-pointer shrink-0 flex items-start justify-center pt-3 transition-colors duration-150 hover:bg-surface-2 after:content-[''] after:absolute after:left-0 after:top-0 after:bottom-0 after:w-0.5 after:bg-accent after:opacity-70"
      >
        <Icon
          name="document"
          className="w-4 h-4 text-text-muted transition-colors duration-150 group-hover:text-accent"
        />
      </button>
      <div
        className={
          "overflow-hidden shrink-0 border-l transition-[width,border-color] duration-[260ms] ease-[cubic-bezier(0.2,0.8,0.2,1)] " +
          (open ? "w-[420px] border-accent" : "w-0 border-border")
        }
      >
        <div className="w-[420px] h-full box-border bg-surface p-4 flex flex-col gap-4">
          <div className="flex justify-end gap-1 shrink-0">
            <button type="button" onClick={onMaximize} aria-label="Büyüt" className={ICON_BTN}>
              <Icon name="maximize" />
            </button>
            <button type="button" onClick={onClose} aria-label="Kapat" className={ICON_BTN}>
              <Icon name="close" />
            </button>
          </div>
          <div className="drawer-panel-body flex-1 overflow-y-auto">{children}</div>
        </div>
      </div>
    </div>
  );
}
