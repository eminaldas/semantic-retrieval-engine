export async function postQuery(payload) {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    const message =
      (data && data.detail && data.detail.message) ||
      (typeof data.detail === "string" ? data.detail : null) ||
      `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function fetchSources() {
  const res = await fetch("/api/sources");
  if (!res.ok) throw new Error(`Sunucu hatasi (${res.status})`);
  return res.json();
}

export async function uploadSource(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/sources/upload", { method: "POST", body: formData });
  const data = await res.json();
  if (!res.ok) {
    const message = (data && data.detail) || `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function fetchUploadStatus(jobId) {
  const res = await fetch(`/api/sources/upload-status/${jobId}`);
  if (!res.ok) throw new Error(`Sunucu hatasi (${res.status})`);
  return res.json();
}

export async function deleteSource(sourceId) {
  const res = await fetch(`/api/sources/${encodeURIComponent(sourceId)}`, { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) {
    const message = (data && data.detail) || `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function fetchSourceContent(sourceId) {
  const res = await fetch(`/api/sources/${encodeURIComponent(sourceId)}/content`);
  const data = await res.json();
  if (!res.ok) {
    const message = (data && data.detail) || `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function fetchConversation() {
  const res = await fetch("/api/conversation");
  if (!res.ok) throw new Error(`Sunucu hatasi (${res.status})`);
  return res.json();
}

export async function deleteConversation() {
  const res = await fetch("/api/conversation", { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) {
    const message = (data && data.detail) || `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function login(password) {
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  const data = await res.json();
  if (!res.ok) {
    const message = (data && data.detail) || `Sunucu hatasi (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export async function logout() {
  const res = await fetch("/api/logout", { method: "POST" });
  if (!res.ok) throw new Error(`Sunucu hatasi (${res.status})`);
  return res.json();
}

export async function fetchSession() {
  const res = await fetch("/api/session");
  if (!res.ok) throw new Error(`Sunucu hatasi (${res.status})`);
  return res.json();
}
