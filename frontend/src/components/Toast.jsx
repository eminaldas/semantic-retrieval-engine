import { useEffect, useState } from "react";

const ICON_LABEL = { success: "✓", error: "✕", info: "i" };
const BORDER_COLOR = { success: "border-l-risk-low", error: "border-l-risk-high", info: "border-l-accent" };

export default function Toast({ kind, message, onDismiss, duration = 3500 }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setShow(true));
    const hideTimer = setTimeout(() => setShow(false), duration);
    const removeTimer = setTimeout(onDismiss, duration + 300);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(hideTimer);
      clearTimeout(removeTimer);
    };
  }, [duration, onDismiss]);

  return (
    <div
      className={
        "flex items-center gap-2 bg-surface-2 border border-border border-l-[3px] rounded-lg px-3 py-2.5 text-[13px] shadow-strong max-w-[320px] transition-[opacity,transform] duration-[220ms] ease-[cubic-bezier(0.2,0.8,0.2,1)] " +
        BORDER_COLOR[kind] +
        (show ? " opacity-100 translate-y-0 scale-100" : " opacity-0 translate-y-1.5 scale-[0.98]")
      }
    >
      <span className="text-xs shrink-0">{ICON_LABEL[kind] || ICON_LABEL.info}</span>
      <span>{message}</span>
    </div>
  );
}
