from __future__ import annotations

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
        self.root.geometry("760x620")

        win_bg = "#ececec"
        text_bg = "#ffffff"
        text_fg = "#101010"
        self.root.configure(bg=win_bg)

        from depression_ml.risk import load_artifacts

        self.model, self.vectorizer, self.scaler, self.thresholds, self.platt = load_artifacts(artifacts_dir)

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

        self.risk_label = tk.Label(result_frame, text="Waiting…", font=_font_heading()[:2] + ("normal",), bg=win_bg, fg=text_fg)
        self.risk_label.pack(pady=4)

        self.prob_label = tk.Label(result_frame, text="", font=_font_body(), bg=win_bg, fg=text_fg)
        self.prob_label.pack(pady=2)

        self.advice_label = tk.Label(
            result_frame, text="", font=_font_body(), wraplength=680, justify=tk.LEFT, bg=win_bg, fg=text_fg
        )
        self.advice_label.pack(pady=8)

        self.lex_label = tk.Label(
            result_frame,
            text="",
            font=_font_body(),
            wraplength=680,
            justify=tk.LEFT,
            bg=win_bg,
            fg="#555555",
        )
        self.lex_label.pack(pady=4)

        body = tk.Frame(self.root, bg=win_bg)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        title = tk.Label(body, text="Depression tendency screening", font=_font_title(), bg=win_bg, fg=text_fg)
        title.pack(pady=10)

        desc = tk.Label(
            body,
            text="Educational demo only — not a medical device. English social-media style text works best.",
            font=_font_body(),
            wraplength=700,
            bg=win_bg,
            fg=text_fg,
        )
        desc.pack(pady=4)

        text_label = tk.Label(body, text="Paste text to analyse:", font=_font_body(), bg=win_bg, fg=text_fg)
        text_label.pack(anchor=tk.W, padx=16, pady=(16, 4))

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

        btn_bar = tk.Frame(body, bg="#2E7D32", cursor="hand2")
        btn_bar.pack(pady=12)
        btn_lbl = tk.Label(
            btn_bar,
            text="Run assessment",
            bg="#2E7D32",
            fg="#ffffff",
            font=_font_heading(),
            padx=28,
            pady=10,
        )
        btn_lbl.pack()
        btn_bar.bind("<Button-1>", lambda _e: self.predict())
        btn_lbl.bind("<Button-1>", lambda _e: self.predict())

    def predict(self) -> None:
        from depression_ml import config
        from depression_ml.emnlp17_signals import extract_emnlp17_detailed
        from depression_ml.preprocess import looks_english, preprocess_text_en
        from depression_ml.risk import depression_probability, risk_tier

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input", "Please enter some text first.")
            return
        if len(text) < config.MIN_TEXT_CHARS:
            messagebox.showwarning(
                "Input",
                f"Text is too short (minimum about {config.MIN_TEXT_CHARS} characters for a stable signal).",
            )
            return
        if not looks_english(text):
            messagebox.showwarning(
                "Input",
                "The text does not look predominantly English. This demo is tuned for English social-media style posts.",
            )
            return
        try:
            clean = preprocess_text_en(text)
            det = extract_emnlp17_detailed(
                clean,
                collect_matches=True,
                match_limit=config.GUI_LEXICON_PREVIEW,
            )
            prob = depression_probability(text, self.model, self.vectorizer, self.scaler, self.platt)
            tier = risk_tier(prob, self.thresholds)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Prediction failed: {exc}")
            return

        if tier == "低风险":
            color = "green"
            advice = "No strong depression cue in this snippet. If you still feel unwell, talking to someone you trust helps."
        elif tier == "中风险":
            color = "darkorange"
            advice = "Some cues resemble distress language. Consider self-care and professional support if symptoms persist."
        else:
            color = "red"
            advice = "Strong distress cues in text. If you or someone else may self-harm, contact local emergency services or a crisis hotline immediately."

        self.risk_label.config(text=f"Risk tier: {tier}", fg=color)
        self.prob_label.config(
            text=f"Model score (depression-positive, Platt-calibrated if trained): {prob:.2%}"
        )
        self.advice_label.config(text=advice, fg=color)

        def _fmt(name: str, xs: list) -> str:
            if not xs:
                return f"{name}: —"
            return f"{name}: " + "; ".join(xs)

        self.lex_label.config(
            text=(
                "Lexicon (EMNLP’17 user_selection, explainability only)\n"
                f"Counts — MH={int(det['emnlp_mh_hits'])}, pos_diag≈{int(det['emnlp_pos_diag'])}, neg_diag≈{int(det['emnlp_neg_diag'])}\n"
                f"{_fmt('MH terms', det['mh_matches'])}\n"
                f"{_fmt('Pos phrases', det['pos_matches'])}\n"
                f"{_fmt('Neg phrases', det['neg_matches'])}"
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
