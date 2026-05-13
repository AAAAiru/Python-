from __future__ import annotations

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


class DepressionDetectorGUI:
    def __init__(self, root: tk.Tk, artifacts_dir: Path):
        self.root = root
        self.artifacts_dir = artifacts_dir
        self.root.title("Depression tendency screening (course demo)")
        self.root.geometry("760x620")

        from depression_ml.risk import load_artifacts

        self.model, self.vectorizer, self.scaler, self.thresholds = load_artifacts(artifacts_dir)

        title = tk.Label(self.root, text="Depression tendency screening", font=("Segoe UI", 16, "bold"))
        title.pack(pady=10)

        desc = tk.Label(
            self.root,
            text="Educational demo only — not a medical device. English social-media style text works best.",
            font=("Segoe UI", 10),
            wraplength=700,
        )
        desc.pack(pady=4)

        text_label = tk.Label(self.root, text="Paste text to analyse:", font=("Segoe UI", 11))
        text_label.pack(anchor=tk.W, padx=16, pady=(16, 4))

        self.text_input = scrolledtext.ScrolledText(self.root, height=14, width=88, font=("Consolas", 10))
        self.text_input.pack(padx=16, pady=4)

        self.predict_btn = tk.Button(
            self.root,
            text="Run assessment",
            command=self.predict,
            bg="#2E7D32",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            width=22,
        )
        self.predict_btn.pack(pady=12)

        result_frame = tk.LabelFrame(self.root, text="Result", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        self.risk_label = tk.Label(result_frame, text="Waiting…", font=("Segoe UI", 13))
        self.risk_label.pack(pady=4)

        self.prob_label = tk.Label(result_frame, text="", font=("Segoe UI", 10))
        self.prob_label.pack(pady=2)

        self.advice_label = tk.Label(result_frame, text="", font=("Segoe UI", 10), wraplength=680, justify=tk.LEFT)
        self.advice_label.pack(pady=8)

    def predict(self) -> None:
        from depression_ml.risk import depression_probability, risk_tier

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input", "Please enter some text first.")
            return
        try:
            prob = depression_probability(text, self.model, self.vectorizer, self.scaler)
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
        self.prob_label.config(text=f"Model score (depression-positive class): {prob:.2%}")
        self.advice_label.config(text=advice, fg=color)


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
