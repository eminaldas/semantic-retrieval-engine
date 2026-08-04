import PanelTabs from "./PanelTabs";
import SourcesPanel from "./SourcesPanel";
import HistoryPanel from "./HistoryPanel";

const TABS = [
  { key: "sources", label: "Kaynaklar", icon: "sources" },
  { key: "history", label: "Geçmiş", icon: "history" },
];

export default function LeftPanel({ turns, activeTab, onTabChange, onPreviewSource, onSelectTurn }) {
  return (
    <div className="p-4 flex flex-col gap-4">
      <PanelTabs tabs={TABS} active={activeTab} onChange={onTabChange} />
      {activeTab === "sources" && <SourcesPanel onPreviewSource={onPreviewSource} />}
      {activeTab === "history" && <HistoryPanel turns={turns} onSelectTurn={onSelectTurn} />}
    </div>
  );
}
