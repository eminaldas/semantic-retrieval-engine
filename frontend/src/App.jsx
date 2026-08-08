import { useEffect, useState } from "react";
import Layout from "./components/Layout";
import LeftPanel from "./components/LeftPanel";
import TopBar from "./components/TopBar";
import Drawer from "./components/Drawer";
import EvidenceDrawerContent from "./components/EvidenceDrawerContent";
import ReadingModal from "./components/ReadingModal";
import QuestionForm from "./components/QuestionForm";
import ChatHistory from "./components/ChatHistory";
import LoginScreen from "./components/LoginScreen";
import { postQuery, fetchSourceContent, fetchConversation, deleteConversation, fetchSession, logout } from "./api";
import { useToast } from "./components/ToastContext.jsx";
import "./index.css";

const TAB_ICONS = { sources: "sources", history: "history" };

let nextTurnId = 1;

export default function App() {
  const [authChecked, setAuthChecked] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [turns, setTurns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [selectedSourceContent, setSelectedSourceContent] = useState(null);
  const [readingModalOpen, setReadingModalOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState("sources");
  const toast = useToast();

  useEffect(() => {
    fetchSession()
      .then((data) => setAuthenticated(data.authenticated))
      .catch(() => setAuthenticated(false))
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    if (!authenticated) return;
    fetchConversation()
      .then((hydrated) => {
        setTurns(hydrated);
        const maxId = hydrated.reduce((max, t) => Math.max(max, t.id), 0);
        nextTurnId = maxId + 1;
      })
      .catch(() => {
        toast.error("Konuşma geçmişi yüklenemedi.");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authenticated]);

  async function handleAsk(question, options = {}) {
    const { turnId, forceAnswer } = options;
    const id = turnId ?? nextTurnId++;
    setTurns((prev) => {
      const pendingTurn = { id, question, pending: true };
      const exists = prev.some((t) => t.id === id);
      return exists ? prev.map((t) => (t.id === id ? pendingTurn : t)) : [...prev, pendingTurn];
    });
    setLoading(true);
    const startedAt = performance.now();
    try {
      const response = await postQuery({ question, force_answer: !!forceAnswer });
      const durationMs = performance.now() - startedAt;
      setTurns((prev) => prev.map((t) => (t.id === id ? { id, question, response, durationMs } : t)));
    } catch (err) {
      setTurns((prev) => prev.map((t) => (t.id === id ? { id, question, error: err.message } : t)));
    } finally {
      setLoading(false);
    }
  }

  function handleForceAnswer(turn) {
    handleAsk(turn.question, { turnId: turn.id, forceAnswer: true });
  }

  async function handleClearConversation() {
    try {
      await deleteConversation();
      setTurns([]);
      setSelectedEvidence(null);
      setSelectedSourceContent(null);
      setDrawerOpen(false);
    } catch {
      toast.error("Konuşma temizlenemedi, tekrar deneyin.");
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // Sessizce yok say - yine de giris ekranina donuyoruz.
    }
    setAuthenticated(false);
  }

  async function handleSelectEvidence(evidence) {
    setSelectedEvidence(evidence);
    setDrawerOpen(true);
    try {
      const chunks = await fetchSourceContent(evidence.source_id);
      setSelectedSourceContent({ source_id: evidence.source_id, chunks });
    } catch {
      toast.error("Kaynağın tamamı alınamadı.");
    }
  }

  function handleSelectTurn(turnId) {
    const el = document.getElementById(`turn-${turnId}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function handlePreviewSource(sourceId) {
    setSelectedEvidence(null);
    setDrawerOpen(true);
    try {
      const chunks = await fetchSourceContent(sourceId);
      setSelectedSourceContent({ source_id: sourceId, chunks });
    } catch {
      toast.error("Kaynak içeriği alınamadı.");
    }
  }

  if (!authChecked) return null;
  if (!authenticated) return <LoginScreen onSuccess={() => setAuthenticated(true)} />;

  return (
    <div className="flex flex-col h-screen">
      <TopBar onClearConversation={handleClearConversation} onLogout={handleLogout} />
      <Layout
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
        sidebarIcon={TAB_ICONS[activeTab]}
        sidebar={
          <LeftPanel
            turns={turns}
            activeTab={activeTab}
            onTabChange={setActiveTab}
            onPreviewSource={handlePreviewSource}
            onSelectTurn={handleSelectTurn}
          />
        }
        main={
          <div className="py-6 flex flex-col gap-4 min-h-full">
            <ChatHistory
              turns={turns}
              selectedSid={selectedEvidence && selectedEvidence.sid}
              onSelectEvidence={handleSelectEvidence}
              onForceAnswer={handleForceAnswer}
            />
            <QuestionForm onSubmit={handleAsk} loading={loading} />
          </div>
        }
        rightPanel={
          <Drawer
            open={drawerOpen}
            onOpen={() => setDrawerOpen(true)}
            onClose={() => setDrawerOpen(false)}
            onMaximize={() => setReadingModalOpen(true)}
          >
            <EvidenceDrawerContent evidence={selectedEvidence} sourceContent={selectedSourceContent} />
          </Drawer>
        }
      />
      <ReadingModal
        open={readingModalOpen}
        onClose={() => setReadingModalOpen(false)}
        evidence={selectedEvidence}
        sourceContent={selectedSourceContent}
      />
    </div>
  );
}
