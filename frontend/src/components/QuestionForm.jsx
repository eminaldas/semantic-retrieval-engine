import { useState } from "react";
import Icon from "./Icon";

export default function QuestionForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState("");

  function handleSubmit(ev) {
    ev.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
    setQuestion("");
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-[700px] w-full mx-auto px-6 box-border pt-2 pb-6">
      <div className="flex items-center gap-1.5 bg-surface rounded-2xl pl-[18px] pr-1.5 py-1.5 shadow-[inset_0_0_0_1px_transparent] transition-shadow duration-150 focus-within:shadow-[inset_0_0_0_1px_var(--color-accent)]">
        <input
          type="text"
          value={question}
          onChange={(ev) => setQuestion(ev.target.value)}
          placeholder="Semantik ağı sorgulayın..."
          disabled={loading}
          className="flex-1 min-w-0 py-2.5 border-0 bg-transparent text-text text-sm outline-none"
        />
        <button
          type="submit"
          disabled={loading}
          aria-label="Gönder"
          className="w-[38px] h-[38px] shrink-0 p-0 border-0 rounded-xl bg-accent text-accent-contrast flex items-center justify-center cursor-pointer transition-[background,transform] duration-150 enabled:hover:bg-accent-strong enabled:active:scale-[0.97] disabled:opacity-50 disabled:cursor-default [&_svg]:w-[18px] [&_svg]:h-[18px]"
        >
          <Icon name="send" />
        </button>
      </div>
    </form>
  );
}
