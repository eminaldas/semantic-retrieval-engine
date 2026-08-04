import Icon from "./Icon";

export default function PanelTabs({ tabs, active, onChange }) {
  return (
    <div className="flex flex-row gap-1 border-b border-border">
      {tabs.map((tab) => {
        const isActive = active === tab.key;
        return (
          <button
            key={tab.key}
            type="button"
            onClick={() => onChange(tab.key)}
            className={
              "group flex-1 flex items-center justify-center gap-2 px-2.5 py-2.5 rounded-t-md font-mono text-xs cursor-pointer transition-colors duration-150 border-b-2 " +
              (isActive
                ? "text-accent-strong border-accent font-semibold"
                : "text-text-muted border-transparent hover:text-text")
            }
          >
            <Icon
              name={tab.icon}
              className="w-4 h-4 shrink-0 transition-transform duration-150 group-hover:scale-110"
            />
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
