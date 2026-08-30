"""Gradio app over the shared backend-neutral inference service."""

from __future__ import annotations

import html
from dataclasses import asdict, dataclass

from PIL import Image

from prooflens.data.transforms import apply_transform, canonical_specs, get_spec
from prooflens.errors import ProofLensError, UserInputError
from prooflens.inference.service import InferenceService, Prediction


@dataclass(frozen=True, slots=True)
class UploadAnalysis:
    clean_image: Image.Image
    transformed_image: Image.Image
    summary: dict[str, object]

    def as_outputs(
        self,
    ) -> tuple[
        Image.Image,
        Image.Image,
        dict[str, float],
        dict[str, float],
        str,
        str,
        str,
        dict[str, object],
    ]:
        clean = self.summary["clean"]
        transformed = self.summary["transformed"]
        if not isinstance(clean, dict) or not isinstance(transformed, dict):
            raise TypeError("analysis summary has invalid prediction values")
        return (
            self.clean_image,
            self.transformed_image,
            {
                "AI-generated": float(clean["probability_ai"]),
                "Authentic": float(clean["probability_real"]),
            },
            {
                "AI-generated": float(transformed["probability_ai"]),
                "Authentic": float(transformed["probability_real"]),
            },
            _verdict_card(self.summary),
            _stability_card(self.summary),
            _provenance(self.summary),
            self.summary,
        )


def analyze_upload(
    image: Image.Image | None,
    condition_id: str,
    service: InferenceService,
) -> UploadAnalysis:
    if image is None:
        raise UserInputError("Upload an image before analysis")
    try:
        clean_image = image.convert("RGB")
        clean_image.load()
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("Upload a decodable image") from error
    spec = get_spec(condition_id)
    transformed_image = apply_transform(clean_image, spec, seed=17)
    stability = service.compare_transform(clean_image, spec, seed=17)
    threshold = service.operating_threshold
    clean_is_ai = stability.clean.probability_ai >= threshold
    transformed_is_ai = stability.transformed.probability_ai >= threshold
    return UploadAnalysis(
        clean_image=clean_image,
        transformed_image=transformed_image,
        summary={
            "condition": condition_id,
            "transform_parameters": dict(spec.parameters),
            "operating_threshold": service.operating_threshold,
            "model_version": stability.clean.model_version,
            "preprocessing_version": stability.clean.preprocessing_version,
            "clean": _prediction_summary(stability.clean, service.operating_threshold),
            "transformed": _prediction_summary(
                stability.transformed, service.operating_threshold
            ),
            "absolute_change": stability.absolute_change,
            "decision": {
                "operating_threshold": threshold,
                "clean_signal": "ai-generated" if clean_is_ai else "authentic",
                "transformed_signal": "ai-generated" if transformed_is_ai else "authentic",
                "verdict_changed": clean_is_ai != transformed_is_ai,
                "stability_rating": _stability_rating(stability.absolute_change),
            },
        },
    )


def _prediction_summary(prediction: Prediction, operating_threshold: float) -> dict[str, object]:
    """Return one prediction with its threshold-relative demo label."""

    return {
        **asdict(prediction),
        "threshold_label": (
            "AI-generated" if prediction.probability_ai >= operating_threshold else "Authentic"
        ),
    }


def create_app(service: InferenceService):
    import gradio as gr

    with gr.Blocks(title="ProofLens") as app:
        with gr.Column(elem_id="prooflens-shell"):
            gr.Markdown(
                "# ProofLens\n"
                "### Image authenticity signals that are tested for robustness"
            )
            gr.Markdown(
                "Research demonstration only — scores indicate model evidence, not forensic proof.",
                elem_classes=["disclaimer"],
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["panel"]):
                    image = gr.Image(
                        type="pil",
                        label="Upload an image",
                        height=360,
                    )
                with gr.Column(scale=2, elem_classes=["panel"]):
                    gr.Markdown("### Robustness check")
                    gr.Markdown(
                        "ProofLens analyzes the original and a controlled transformation to show "
                        "whether the result survives ordinary image processing."
                    )
                    condition = gr.Dropdown(
                        choices=[spec.condition_id for spec in canonical_specs()],
                        value="jpeg_q30",
                        label="Transformation",
                    )
                    analyze = gr.Button("Analyze image", variant="primary", size="lg")

            gr.Markdown("## Result")
            verdict = gr.HTML(_empty_verdict_card())
            stability_card = gr.HTML(_empty_stability_card())
            provenance = gr.Markdown(
                "Upload an image to see model and preprocessing provenance.",
                elem_classes=["provenance"],
            )

            with gr.Accordion("Inspect images and probabilities", open=True):
                with gr.Row():
                    clean_output = gr.Image(label="Original image")
                    transformed_output = gr.Image(label="Transformed image")
                with gr.Row():
                    clean_probabilities = gr.Label(label="Original probabilities")
                    transformed_probabilities = gr.Label(label="Transformed probabilities")

            with gr.Accordion("Technical details", open=False):
                summary = gr.JSON(label="Prediction record")

            with gr.Accordion("Model limitations", open=False):
                gr.Markdown(
                    "- Scores may shift for generators or image domains absent from training.\n"
                    "- Compression, resizing, screenshots, and unusual content can affect results.\n"
                    "- A high score does not establish authorship or manipulation.\n"
                    "- Fixture-demo artifacts validate the workflow only and are not a real-world detector."
                )

        def callback(uploaded: Image.Image | None, selected: str):
            try:
                return analyze_upload(uploaded, selected, service).as_outputs()
            except ProofLensError as error:
                message = html.escape(str(error))
                return (
                    None,
                    None,
                    {},
                    {},
                    (
                        '<div class="result-card neutral">'
                        "<strong>Cannot analyze image</strong>"
                        f"<span>{message}</span></div>"
                    ),
                    _empty_stability_card(),
                    "Check the upload and try again.",
                    {"error": str(error)},
                )

        analyze.click(
            fn=callback,
            inputs=[image, condition],
            outputs=[
                clean_output,
                transformed_output,
                clean_probabilities,
                transformed_probabilities,
                verdict,
                stability_card,
                provenance,
                summary,
            ],
        )
    app.prooflens_css = _APP_CSS
    return app


def _verdict_card(summary: dict[str, object]) -> str:
    clean, _, decision = _summary_parts(summary)
    probability_ai = float(clean["probability_ai"])
    confidence = float(clean["confidence"])
    threshold = float(decision["operating_threshold"])
    ai_signal = decision["clean_signal"] == "ai-generated"
    signal = "AI-generated signal" if ai_signal else "Authentic signal"
    tone = "ai" if ai_signal else "authentic"
    confidence_label = _confidence_label(confidence)
    return (
        f'<div class="result-card {tone}">'
        '<span class="eyebrow">Primary result</span>'
        f"<strong>{signal}</strong>"
        f'<span class="result-copy">AI probability {probability_ai:.1%} · '
        f"Decision threshold {threshold:.1%}</span>"
        f'<span class="pill">{confidence_label} · {confidence:.1%} confidence</span>'
        "</div>"
    )


def _stability_card(summary: dict[str, object]) -> str:
    _, _, decision = _summary_parts(summary)
    absolute_change = float(summary["absolute_change"])
    condition = html.escape(str(summary["condition"]).replace("_", " "))
    rating = html.escape(str(decision["stability_rating"]))
    changed = bool(decision["verdict_changed"])
    tone = "warning" if changed or absolute_change > 0.10 else "stable"
    change_note = "The decision changed after transformation." if changed else "The decision held."
    return (
        f'<div class="result-card {tone}">'
        '<span class="eyebrow">Robustness</span>'
        f"<strong>{rating}</strong>"
        f'<span class="result-copy">{condition} changed the AI probability by '
        f"{absolute_change:.1%}. {change_note}</span>"
        "</div>"
    )


def _provenance(summary: dict[str, object]) -> str:
    clean, transformed, _ = _summary_parts(summary)
    model_version = str(clean["model_version"])
    preprocessing_version = str(clean["preprocessing_version"])
    elapsed = float(clean["inference_ms"]) + float(transformed["inference_ms"])
    return (
        f"**Model:** `{model_version}` · **Preprocessing:** `{preprocessing_version}` · "
        f"**Two-pass inference:** `{elapsed:.1f} ms`"
    )


def _summary_parts(
    summary: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    clean = summary.get("clean")
    transformed = summary.get("transformed")
    decision = summary.get("decision")
    if not all(isinstance(value, dict) for value in (clean, transformed, decision)):
        raise TypeError("analysis summary is missing prediction presentation fields")
    return clean, transformed, decision  # type: ignore[return-value]


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.65:
        return "Moderate confidence"
    return "Low confidence"


def _stability_rating(absolute_change: float) -> str:
    if absolute_change <= 0.02:
        return "Stable under transformation"
    if absolute_change <= 0.10:
        return "Moderate probability shift"
    return "Sensitive to transformation"


def _empty_verdict_card() -> str:
    return (
        '<div class="result-card neutral"><span class="eyebrow">Primary result</span>'
        "<strong>Waiting for an image</strong>"
        '<span class="result-copy">Upload an image and select Analyze image.</span></div>'
    )


def _empty_stability_card() -> str:
    return (
        '<div class="result-card neutral"><span class="eyebrow">Robustness</span>'
        "<strong>No comparison yet</strong>"
        '<span class="result-copy">A transformed comparison will appear here.</span></div>'
    )


_APP_CSS = """
#prooflens-shell { max-width: 1120px; margin: 0 auto; padding: 20px 12px 48px; }
.panel { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 18px; padding: 18px; box-shadow: 0 12px 30px rgba(15, 23, 42, .06); }
.disclaimer { border-left: 4px solid #f59e0b; padding: 10px 14px; background: rgba(245, 158, 11, .10);
  border-radius: 8px; margin-bottom: 12px; }
.result-card { border-radius: 18px; padding: 20px 22px; margin: 8px 0; display: flex;
  flex-direction: column; gap: 6px; border: 1px solid transparent; }
.result-card strong { font-size: 1.45rem; line-height: 1.2; }
.result-card .eyebrow { font-size: .75rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; opacity: .72; }
.result-card .result-copy { font-size: .95rem; opacity: .88; }
.result-card .pill { align-self: flex-start; margin-top: 6px; padding: 5px 10px; border-radius: 999px;
  background: rgba(255, 255, 255, .62); font-size: .82rem; font-weight: 650; }
.result-card.ai { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border-color: #fda4af; color: #881337; }
.result-card.authentic { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border-color: #6ee7b7; color: #064e3b; }
.result-card.stable { background: linear-gradient(135deg, #eff6ff, #dbeafe); border-color: #93c5fd; color: #1e3a8a; }
.result-card.warning { background: linear-gradient(135deg, #fffbeb, #fef3c7); border-color: #fcd34d; color: #78350f; }
.result-card.neutral { background: var(--block-background-fill); border-color: var(--border-color-primary); }
.provenance { font-size: .9rem; opacity: .82; }
"""
