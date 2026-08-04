import Toast from "./Toast";

export default function ToastStack({ toasts, onDismiss }) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed right-4 bottom-4 flex flex-col gap-2 z-50 items-end">
      {toasts.map((t) => (
        <Toast key={t.id} kind={t.kind} message={t.message} onDismiss={() => onDismiss(t.id)} />
      ))}
    </div>
  );
}
