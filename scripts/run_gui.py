from __future__ import annotations

import json
import platform
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext


def _ensure_src_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _font_title() -> tuple[str, int, str]:
    if platform.system() == "Darwin":
        return ("Helvetica Neue", 16, "bold")
    return ("Segoe UI", 16, "bold")


def _font_heading() -> tuple[str, int, str]:
    if platform.system() == "Darwin":
        return ("Helvetica Neue", 11, "bold")
    return ("Segoe UI", 11, "bold")


def _font_body() -> tuple[str, int]:
    if platform.system() == "Darwin":
        return ("Helvetica Neue", 10)
    return ("Segoe UI", 10)


def _font_mono() -> tuple[str, int]:
    if platform.system() == "Darwin":
        return ("Menlo", 10)
    return ("Consolas", 10)


class DepressionDetectorGUI:
    def __init__(self, root: tk.Tk, artifacts_dir: Path):
        self.root = root
        self.artifacts_dir = artifacts_dir
        self.root.title("Depression tendency screening (course demo)")
        self.root.geometry("780x680")

        win_bg = "#ececec"
        text_bg = "#ffffff"
        text_fg = "#101010"
        self._win_bg = win_bg
        self._text_fg = text_fg
        self.root.configure(bg=win_bg)

        from depression_ml.risk import load_artifacts

        self.model, self.vectorizer, self.scaler, self.thresholds, self.platt = load_artifacts(artifacts_dir)
        metadata_path = artifacts_dir / "model_metadata.json"
        self.model_version = "unknown"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.model_version = str(metadata.get("model_version") or "unknown")
            except (OSError, ValueError, TypeError):
                pass

        result_frame = tk.LabelFrame(
            self.root,
            text="Result",
            font=_font_heading(),
            padx=10,
            pady=10,
            bg=win_bg,
            fg=text_fg,
            highlightbackground="#c0c0c0",
        )
        result_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=8)

        result_toolbar = tk.Frame(result_frame, bg=win_bg)
        result_toolbar.pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            result_toolbar,
            text="After each test, reset here — no need to restart from terminal:",
            font=_font_body(),
            bg=win_bg,
            fg="#555555",
        ).pack(side=tk.LEFT)
        reset_bar = tk.Frame(result_toolbar, bg="#1565C0", cursor="hand2")
        reset_bar.pack(side=tk.RIGHT)
        reset_lbl = tk.Label(
            reset_bar,
            text="一键归零",
            bg="#1565C0",
            fg="#ffffff",
            font=_font_heading(),
            padx=18,
            pady=6,
        )
        reset_lbl.pack()
        reset_bar.bind("<Button-1>", lambda _e: self.reset_demo())
        reset_lbl.bind("<Button-1>", lambda _e: self.reset_demo())

        self.risk_label = tk.Label(result_frame, text="Waiting…", font=_font_heading()[:2] + ("normal",), bg=win_bg, fg=text_fg)
        self.risk_label.pack(pady=4)

        self.conf_label = tk.Label(result_frame, text="", font=_font_body(), bg=win_bg, fg="#555555")
        self.conf_label.pack(pady=2)

        self.prob_label = tk.Label(result_frame, text="", font=_font_body(), bg=win_bg, fg=text_fg)
        self.prob_label.pack(pady=2)

        self.advice_label = tk.Label(
            result_frame, text="", font=_font_body(), wraplength=700, justify=tk.LEFT, bg=win_bg, fg=text_fg
        )
        self.advice_label.pack(pady=6)

        self.note_label = tk.Label(
            result_frame, text="", font=_font_body(), wraplength=700, justify=tk.LEFT, bg=win_bg, fg="#666666"
        )
        self.note_label.pack(pady=2)

        self.lex_label = tk.Label(
            result_frame,
            text="",
            font=_font_body(),
            wraplength=700,
            justify=tk.LEFT,
            bg=win_bg,
            fg="#555555",
        )
        self.lex_label.config(text="Run an assessment to see lexicon explainability cues here.")
        self.lex_label.pack(pady=4)

        body = tk.Frame(self.root, bg=win_bg)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        title = tk.Label(body, text="Depression tendency screening", font=_font_title(), bg=win_bg, fg=text_fg)
        title.pack(pady=10)

        desc = tk.Label(
            body,
            text=(
                "Educational demo only — not a medical device. "
                "Designed for English social-media style text; outputs are not diagnoses. "
                f"Model version: {self.model_version}. "
                "Paste text → Run assessment → 一键归零 for the next case."
            ),
            font=_font_body(),
            wraplength=720,
            bg=win_bg,
            fg=text_fg,
        )
        desc.pack(pady=4)

        input_row = tk.Frame(body, bg=win_bg)
        input_row.pack(fill=tk.X, padx=16, pady=(16, 4))
        text_label = tk.Label(input_row, text="Paste text to analyse:", font=_font_body(), bg=win_bg, fg=text_fg)
        text_label.pack(side=tk.LEFT)

        self.text_input = scrolledtext.ScrolledText(
            body,
            height=14,
            width=88,
            font=_font_mono(),
            bg=text_bg,
            fg=text_fg,
            insertbackground=text_fg,
            highlightthickness=1,
            highlightbackground="#b0b0b0",
            relief=tk.FLAT,
        )
        self.text_input.pack(padx=16, pady=4, fill=tk.BOTH, expand=True)

        run_bar = tk.Frame(input_row, bg="#2E7D32", cursor="hand2")
        run_bar.pack(side=tk.RIGHT)
        run_lbl = tk.Label(
            run_bar,
            text="Run assessment",
            bg="#2E7D32",
            fg="#ffffff",
            font=_font_heading(),
            padx=20,
            pady=6,
        )
        run_lbl.pack()
        run_bar.bind("<Button-1>", lambda _e: self.predict())
        run_lbl.bind("<Button-1>", lambda _e: self.predict())

        self.root.bind("<Escape>", lambda _e: self.reset_demo())
        self.root.bind("<Control-Return>", lambda _e: self.predict())

    def reset_demo(self) -> None:
        """Clear input and restore the result panel to the initial waiting state."""
        self.text_input.delete("1.0", tk.END)
        self.risk_label.config(text="Waiting…", fg=self._text_fg)
        self.conf_label.config(text="")
        self.prob_label.config(text="")
        self.advice_label.config(text="", fg=self._text_fg)
        self.note_label.config(text="")
        self.lex_label.config(
            text="Run an assessment to see lexicon explainability cues here."
        )
        self.text_input.focus_set()

    def predict(self) -> None:
        from depression_ml import config
        from depression_ml.preprocess import looks_english
        from depression_ml.risk import assess_text

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input", "Please enter some text first.")
            return
        if len(text) < config.MIN_TEXT_CHARS:
            messagebox.showwarning(
                "Input",
                f"Text is too short (need at least {config.MIN_TEXT_CHARS} characters).",
            )
            return
        if not looks_english(text):
            messagebox.showwarning(
                "Input",
                "The text does not look predominantly English. This demo is tuned for English social-media style posts.",
            )
            return

        if len(text) < config.MIN_TEXT_CHARS_SOFT:
            if not messagebox.askyesno(
                "Short text",
                f"This snippet is under {config.MIN_TEXT_CHARS_SOFT} characters. "
                "Scores are unreliable for very short posts. Continue anyway?",
            ):
                return

        try:
            result = assess_text(
                text,
                self.artifacts_dir,
                model=self.model,
                vectorizer=self.vectorizer,
                scaler=self.scaler,
                thresholds=self.thresholds,
                platt=self.platt,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Prediction failed: {exc}")
            return

        tier = result.tier
        prob = result.prob
        det = result.lex

        if tier == "低风险":
            color = "green"
            advice = "Displayed text-alert tier is low. This does not rule out distress or replace professional assessment."
        elif tier == "中风险":
            color = "darkorange"
            advice = "Some language resembles distress patterns. Consider support if these feelings persist."
        else:
            color = "red"
            advice = "Strong distress patterns were detected in the text."
            if "explicit_crisis_language" in result.flags:
                advice += " Contact local emergency services or a crisis hotline immediately if anyone may self-harm."
            else:
                advice += " Consider timely support from a trusted person or qualified professional."

        if result.confidence == "低":
            advice += " (Short or sparse text — treat the score as indicative only.)"

        self.risk_label.config(text=f"Risk tier: {tier}", fg=color)
        self.conf_label.config(
            text=(
                f"Input adequacy (length-based): {result.confidence}"
                f"  |  {result.word_count} words, {result.char_len} chars"
            )
        )

        model_note = ""
        if result.model_tier != tier:
            model_note = f"  |  model-only tier was {result.model_tier}"
        self.prob_label.config(
            text=(
                f"Model-positive score (not medical probability): {prob:.2%}"
                f"{model_note}"
                f"  |  VADER sentiment: {result.sentiment_compound:+.2f}"
            )
        )
        self.advice_label.config(text=advice, fg=color if tier != "中风险" else "#8B4513")

        note = result.flag_notes_zh
        self.note_label.config(text=note if note else "")

        def _fmt(name: str, xs: list) -> str:
            if not xs:
                return f"{name}: —"
            return f"{name}: " + "; ".join(xs)

        self.lex_label.config(
            text=(
                "Lexicon evidence (Georgetown emnlp17-depression; not a causal explanation)\n"
                f"Counts — MH={int(det['emnlp_mh_hits'])}, pos_diag≈{int(det['emnlp_pos_diag'])}, neg_diag≈{int(det['emnlp_neg_diag'])}, "
                f"sub_word={int(det['emnlp_subreddit_word_hits'])}, sub_r={int(det['emnlp_subreddit_r_hits'])}\n"
                f"{_fmt('MH terms', det['mh_matches'])}\n"
                f"{_fmt('Pos phrases', det['pos_matches'])}\n"
                f"{_fmt('Neg phrases', det['neg_matches'])}\n"
                f"{_fmt('Subreddits', det['subreddit_matches'])}"
            )
        )


def main() -> None:
    root_dir = _ensure_src_on_path()
    artifacts = root_dir / "artifacts"
    if not (artifacts / "depression_model.pkl").exists():
        print("Artifacts not found. Run: python scripts/run_train.py")
        sys.exit(1)
    root = tk.Tk()
    DepressionDetectorGUI(root, artifacts)
    root.mainloop()


if __name__ == "__main__":
    main()
