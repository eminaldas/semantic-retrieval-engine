import { useState, useEffect, useRef } from "react";
import SourceRow from "./SourceRow";
import { fetchSources, uploadSource, fetchUploadStatus, deleteSource } from "../api";
import { useToast } from "./ToastContext.jsx";

export default function SourcesPanel({ onPreviewSource }) {
  const [sources, setSources] = useState([]);
  const [job, setJob] = useState(null);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState(null);
  const pollRef = useRef(null);
  const toast = useToast();

  async function loadSources() {
    try {
      const data = await fetchSources();
      setSources(data);
    } catch {
      toast.error("Kaynaklar listelenemedi.");
    }
  }

  useEffect(() => {
    loadSources();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return undefined;
    pollRef.current = setInterval(async () => {
      try {
        const status = await fetchUploadStatus(job.job_id);
        setJob((prev) => (prev && prev.job_id === job.job_id ? { ...prev, ...status } : prev));
        if (status.status !== "running") {
          clearInterval(pollRef.current);
          if (status.status === "done") {
            loadSources();
            toast.success(`${status.source_id || job.source_id} başarıyla yüklendi.`);
          }
        }
      } catch {
        clearInterval(pollRef.current);
        toast.error("Yükleme durumu kontrol edilemedi.");
      }
    }, 1500);
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job && job.job_id, job && job.status]);

  async function handleFileSelected(file) {
    if (job && job.status === "running") return;
    try {
      const { job_id, source_id } = await uploadSource(file);
      setJob({
        job_id, source_id, status: "running", step: 0, total_steps: 5,
        step_name: "Başlatılıyor", error_message: null,
      });
    } catch (err) {
      setJob({
        job_id: "upload-error", source_id: file.name, status: "error",
        step: 0, total_steps: 5, step_name: "", error_message: err.message,
      });
    }
  }

  function handleDrop(ev) {
    ev.preventDefault();
    const file = ev.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  }

  async function handleDelete(sourceId) {
    try {
      await deleteSource(sourceId);
      setConfirmingDeleteId(null);
      loadSources();
      toast.success(`${sourceId} silindi.`);
    } catch (err) {
      setConfirmingDeleteId(null);
      toast.error(`Silme başarısız: ${err.message}`);
    }
  }

  const jobSourceShown = job && job.status !== "done" && !sources.some((s) => s.source_id === job.source_id);

  return (
    <div className="flex flex-col gap-4">
      <label
        onDragOver={(ev) => ev.preventDefault()}
        onDrop={handleDrop}
        className="border border-dashed border-border rounded-[10px] py-6 px-4 text-center text-[13px] text-text-muted cursor-pointer transition-colors duration-150 hover:border-accent"
      >
        <input
          type="file"
          className="hidden"
          accept=".txt,.md,.pdf,.docx,.html,.htm"
          onChange={(ev) => ev.target.files[0] && handleFileSelected(ev.target.files[0])}
          disabled={job && job.status === "running"}
        />
        Dosya seç veya sürükle-bırak
      </label>
      <ul className="list-none m-0 p-0 flex flex-col gap-px">
        {sources.map((s) => (
          <SourceRow
            key={s.source_id}
            source={s}
            job={job && job.source_id === s.source_id ? job : null}
            confirming={confirmingDeleteId === s.source_id}
            onDeleteClick={() => setConfirmingDeleteId(s.source_id)}
            onCancelDelete={() => setConfirmingDeleteId(null)}
            onConfirmDelete={() => handleDelete(s.source_id)}
            onPreview={onPreviewSource}
          />
        ))}
        {jobSourceShown && (
          <SourceRow
            source={{ source_id: job.source_id, chunk_count: 0, entity_count: 0, relation_count: 0 }}
            job={job}
            confirming={false}
            onDeleteClick={() => {}}
            onCancelDelete={() => {}}
            onConfirmDelete={() => {}}
            onPreview={() => {}}
          />
        )}
      </ul>
    </div>
  );
}
