from backend.query import step5d_conversation_memory as conv


class TestLoadHistory:
    def test_missing_file_returns_empty_list(self, tmp_path):
        missing = tmp_path / "history.jsonl"
        assert conv.load_history(str(missing)) == []

    def test_parses_jsonl_lines_and_skips_blanks(self, tmp_path):
        history_file = tmp_path / "history.jsonl"
        history_file.write_text(
            '{"turn_id": 1, "question": "a"}\n'
            "\n"
            '{"turn_id": 2, "question": "b"}\n',
            encoding="utf-8",
        )
        turns = conv.load_history(str(history_file))
        assert len(turns) == 2
        assert turns[0]["turn_id"] == 1
        assert turns[1]["question"] == "b"


class TestLoadSummaryState:
    def test_missing_file_returns_default(self, tmp_path):
        missing = tmp_path / "summary.json"
        assert conv.load_summary_state(str(missing)) == {"summary_text": "", "summarized_through_turn_id": 0}

    def test_reads_existing_state(self, tmp_path):
        summary_file = tmp_path / "summary.json"
        summary_file.write_text('{"summary_text": "ozet", "summarized_through_turn_id": 3}', encoding="utf-8")
        assert conv.load_summary_state(str(summary_file)) == {"summary_text": "ozet", "summarized_through_turn_id": 3}

    def test_corrupt_json_returns_default(self, tmp_path):
        summary_file = tmp_path / "summary.json"
        summary_file.write_text("{not valid json", encoding="utf-8")
        assert conv.load_summary_state(str(summary_file)) == {"summary_text": "", "summarized_through_turn_id": 0}


class TestNextTurnId:
    def test_empty_history_returns_one(self):
        assert conv.next_turn_id([]) == 1

    def test_returns_max_plus_one(self):
        history = [{"turn_id": 1}, {"turn_id": 3}, {"turn_id": 2}]
        assert conv.next_turn_id(history) == 4


class TestSelectWindowTurns:
    def test_returns_turns_after_summarized_through_id_in_order(self):
        history = [{"turn_id": 3}, {"turn_id": 1}, {"turn_id": 2}, {"turn_id": 5}]
        result = conv.select_window_turns(history, summarized_through_turn_id=2)
        assert [t["turn_id"] for t in result] == [3, 5]

    def test_empty_history_returns_empty_list(self):
        assert conv.select_window_turns([], summarized_through_turn_id=0) == []


class TestBuildConversationContext:
    def test_no_summary_no_turns_returns_empty_string(self):
        assert conv.build_conversation_context("", []) == ""

    def test_formats_summary_and_turns(self):
        window_turns = [
            {"question": "PostgreSQL neden shared buffer kullanir?", "answer_text": "Performans icin [S1]."},
        ]
        result = conv.build_conversation_context("Kullanici PostgreSQL sordu.", window_turns)
        assert "Onceki konusmanin ozeti: Kullanici PostgreSQL sordu." in result
        assert "S: PostgreSQL neden shared buffer kullanir?" in result
        assert "C: Performans icin [S1]." in result

    def test_only_turns_no_summary(self):
        window_turns = [{"question": "soru", "answer_text": "cevap"}]
        result = conv.build_conversation_context("", window_turns)
        assert "Onceki konusmanin ozeti" not in result
        assert "S: soru" in result


class TestTurnsToFold:
    def test_within_window_returns_empty_list(self):
        window_turns = [{"turn_id": i} for i in range(1, 4)]
        assert conv.turns_to_fold(window_turns, window_size=5) == []

    def test_exceeds_window_returns_oldest_first(self):
        window_turns = [{"turn_id": i} for i in range(1, 8)]
        result = conv.turns_to_fold(window_turns, window_size=5)
        assert [t["turn_id"] for t in result] == [1, 2]


class TestAppendTurn:
    def test_appends_jsonl_line(self, tmp_path):
        history_file = tmp_path / "history.jsonl"
        conv.append_turn(str(history_file), {"turn_id": 1, "question": "a"})
        conv.append_turn(str(history_file), {"turn_id": 2, "question": "b"})
        turns = conv.load_history(str(history_file))
        assert [t["turn_id"] for t in turns] == [1, 2]


class TestSaveSummaryState:
    def test_overwrites_existing_state(self, tmp_path):
        summary_file = tmp_path / "summary.json"
        conv.save_summary_state(str(summary_file), "eski ozet", 1)
        conv.save_summary_state(str(summary_file), "yeni ozet", 3)
        assert conv.load_summary_state(str(summary_file)) == {"summary_text": "yeni ozet", "summarized_through_turn_id": 3}


class TestClearConversation:
    def test_removes_existing_files(self, tmp_path):
        history_file = tmp_path / "history.jsonl"
        summary_file = tmp_path / "summary.json"
        history_file.write_text('{"turn_id": 1}\n', encoding="utf-8")
        summary_file.write_text('{"summary_text": "x", "summarized_through_turn_id": 1}', encoding="utf-8")
        conv.clear_conversation(str(history_file), str(summary_file))
        assert not history_file.exists()
        assert not summary_file.exists()

    def test_missing_files_does_not_raise(self, tmp_path):
        history_file = tmp_path / "history.jsonl"
        summary_file = tmp_path / "summary.json"
        conv.clear_conversation(str(history_file), str(summary_file))
