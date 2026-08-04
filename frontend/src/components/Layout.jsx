import Icon from "./Icon";

const SIDEBAR_WIDTH = 420;
const COLLAPSED_WIDTH = 32;

export default function Layout({ sidebar, main, rightPanel, sidebarCollapsed, onToggleSidebar, sidebarIcon }) {
  return (
    <div className="flex flex-1 min-h-0">
      <div
        className="relative shrink-0 flex flex-col overflow-hidden border-r border-border transition-[width] duration-200 ease-in-out"
        style={{ width: sidebarCollapsed ? COLLAPSED_WIDTH : SIDEBAR_WIDTH }}
      >
        <div className="flex justify-end p-1 shrink-0">
          <button
            type="button"
            className="w-5 h-5 border border-border rounded text-text-muted bg-surface cursor-pointer text-[11px] leading-none"
            onClick={onToggleSidebar}
            aria-label={sidebarCollapsed ? "Kaynaklar panelini aç" : "Kaynaklar panelini kapat"}
          >
            {sidebarCollapsed ? "»" : "«"}
          </button>
        </div>
        {sidebarCollapsed ? (
          <div
            className="group flex-1 flex flex-col items-center pt-1 cursor-pointer"
            onClick={onToggleSidebar}
          >
            <Icon
              name={sidebarIcon}
              className="w-[18px] h-[18px] text-text-muted transition-colors duration-150 group-hover:text-accent"
            />
          </div>
        ) : (
          <div className="layout-sidebar-content flex-1 overflow-y-auto">{sidebar}</div>
        )}
      </div>

      <div className="layout-main flex-1 overflow-y-auto min-w-0">{main}</div>

      <div className="flex shrink-0 h-full">{rightPanel}</div>
    </div>
  );
}
