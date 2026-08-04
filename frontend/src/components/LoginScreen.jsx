import { useState } from "react";
import Logo from "./Logo";
import { login } from "../api";

export default function LoginScreen({ onSuccess }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(ev) {
    ev.preventDefault();
    if (!password.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      await login(password);
      onSuccess();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex items-center justify-center h-screen bg-bg">
      <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4 w-full max-w-[320px] px-6">
        <Logo />
        <input
          type="password"
          value={password}
          onChange={(ev) => setPassword(ev.target.value)}
          placeholder="Şifre"
          autoFocus
          disabled={loading}
          className="w-full bg-surface text-text text-sm rounded-[10px] px-4 py-2.5 outline-none shadow-[inset_0_0_0_1px_var(--color-border)] transition-shadow duration-150 focus:shadow-[inset_0_0_0_1px_var(--color-accent)]"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-accent text-accent-contrast text-sm font-semibold rounded-[10px] px-4 py-2.5 cursor-pointer transition-[background,transform] duration-150 enabled:hover:bg-accent-strong enabled:active:scale-[0.97] disabled:opacity-50 disabled:cursor-default"
        >
          {loading ? "..." : "Giriş Yap"}
        </button>
        {error && <p className="text-risk-high text-[13px] m-0">{error}</p>}
      </form>
    </div>
  );
}
