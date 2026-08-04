import { useEffect, useState } from "react";

const PHRASES = ["Kaynaklar taranıyor", "Kanıtlar değerlendiriliyor", "Yanıt oluşturuluyor"];

export default function LoadingIndicator() {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setIndex((i) => (i + 1) % PHRASES.length);
        setVisible(true);
      }, 250);
    }, 1400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-0.5 py-1 px-0.5 text-text-muted text-[13px]">
      <span className="transition-opacity duration-250" style={{ opacity: visible ? 1 : 0 }}>
        {PHRASES[index]}
      </span>
      <span className="w-1 h-1 rounded-full bg-text-muted ml-[3px] animate-dot-pulse" />
      <span className="w-1 h-1 rounded-full bg-text-muted ml-[3px] animate-dot-pulse" style={{ animationDelay: "0.15s" }} />
      <span className="w-1 h-1 rounded-full bg-text-muted ml-[3px] animate-dot-pulse" style={{ animationDelay: "0.3s" }} />
    </div>
  );
}
