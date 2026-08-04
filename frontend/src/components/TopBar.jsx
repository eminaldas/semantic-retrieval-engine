import Logo from "./Logo";
import Icon from "./Icon";

export default function TopBar({ onClearConversation, onLogout }) {
  return (
    <header className="shrink-0 flex items-center justify-between gap-4 px-6 h-[60px] border-b border-border">
      <Logo />
      <div className="flex items-center gap-2">
        <button
          className="border border-border bg-surface text-text rounded-[10px] px-3 py-1.5 text-[13px] cursor-pointer transition-[border-color,transform] duration-150 hover:border-accent active:scale-[0.97]"
          onClick={onClearConversation}
        >
          Konuşmayı temizle
        </button>
        <button
          type="button"
          aria-label="Çıkış yap"
          onClick={onLogout}
          className="w-9 h-9 border border-border bg-surface text-text-muted rounded-[10px] flex items-center justify-center cursor-pointer transition-colors duration-150 hover:border-accent hover:text-text [&_svg]:w-4 [&_svg]:h-4"
        >
          <Icon name="logout" />
        </button>
      </div>
    </header>
  );
}
