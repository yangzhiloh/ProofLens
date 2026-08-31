"""Gradio app over the shared backend-neutral inference service."""

from __future__ import annotations

import html
from dataclasses import asdict, dataclass

from PIL import Image

from prooflens.data.transforms import apply_transform, get_spec
from prooflens.errors import ProofLensError, UserInputError
from prooflens.inference.processing_signals import ProcessingAssessment, assess_processing
from prooflens.inference.service import InferenceService, Prediction


_TRANSFORMATION_LABELS = {
    "jpeg_q90": "JPEG compression — Light (quality 90)",
    "jpeg_q70": "JPEG compression — Medium (quality 70)",
    "jpeg_q50": "JPEG compression — Strong (quality 50)",
    "jpeg_q30": "JPEG compression — Heavy (quality 30)",
    "blur_s0.5": "Blur — Very light (strength 0.5)",
    "blur_s1.0": "Blur — Medium (strength 1.0)",
    "blur_s2.0": "Blur — Strong (strength 2.0)",
    "resize_x0.5": "Resize — Half resolution (50%)",
    "resize_x0.25": "Resize — Quarter resolution (25%)",
    "noise_s0.02": "Noise — Light (2%)",
    "noise_s0.05": "Noise — Medium (5%)",
    "noise_s0.10": "Noise — Strong (10%)",
    "color_jitter_20": "Color adjustment — Moderate (20%)",
    "center_crop_80": "Center crop — Keep middle 80%",
}


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


@dataclass(frozen=True, slots=True)
class DirectAnalysis:
    image: Image.Image
    prediction: Prediction
    assessment: ProcessingAssessment
    operating_threshold: float

    def as_outputs(self) -> tuple[Image.Image, dict[str, float], str, str, str, dict[str, object]]:
        prediction = _prediction_summary(self.prediction, self.operating_threshold)
        summary = {
            "prediction": prediction,
            "operating_threshold": self.operating_threshold,
            "processing_assessment": self.assessment.to_dict(),
        }
        return (
            self.image,
            {
                "AI-generated": self.prediction.probability_ai,
                "Authentic": self.prediction.probability_real,
            },
            _direct_verdict_card(self.prediction, self.operating_threshold),
            _processing_card(self.assessment),
            _direct_provenance(self.prediction),
            summary,
        )


def analyze_single_upload(
    image: Image.Image | None,
    service: InferenceService,
) -> DirectAnalysis:
    """Analyze the uploaded image exactly as received, without adding another transformation."""

    if image is None:
        raise UserInputError("Upload an image before analysis")
    clean_image = _decode_upload(image)
    return DirectAnalysis(
        image=clean_image,
        prediction=service.predict(clean_image),
        assessment=assess_processing(image),
        operating_threshold=service.operating_threshold,
    )


def analyze_upload(
    image: Image.Image | None,
    condition_id: str,
    service: InferenceService,
) -> UploadAnalysis:
    clean_image = _decode_upload(image)
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

    with gr.Blocks(title="ProofLens", css=_APP_CSS) as app:
        with gr.Column(elem_id="prooflens-shell"):
            gr.Markdown(
                "# ProofLens\n"
                "### Authentic versus AI-generated image analysis"
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
                    gr.Markdown("### Analyze as received")
                    gr.Markdown(
                        "ProofLens accepts images that may already have been compressed, blurred, "
                        "cropped, recolored, or resized. It does not alter the upload before scoring."
                    )
                    analyze = gr.Button("Analyze picture", variant="primary", size="lg")

            gr.Markdown("## Result")
            verdict = gr.HTML(_empty_verdict_card())
            processing_card = gr.HTML(_empty_processing_card())
            provenance = gr.Markdown(
                "Upload an image to see model and preprocessing provenance.",
                elem_classes=["provenance"],
            )

            with gr.Accordion("Inspect image and probabilities", open=True):
                with gr.Row():
                    image_output = gr.Image(label="Analyzed image")
                    probabilities = gr.Label(label="Probabilities")

            with gr.Accordion("Technical details", open=False):
                summary = gr.JSON(label="Prediction record")

            with gr.Accordion("Model limitations", open=False):
                gr.Markdown(
                    "- Scores may shift for generators or image domains absent from training.\n"
                    "- Compression, resizing, screenshots, and unusual content can affect results.\n"
                    "- A high score does not establish authorship or manipulation.\n"
                    "- Fixture-demo artifacts validate the workflow only and are not a real-world detector."
                )

        def callback(uploaded: Image.Image | None):
            try:
                return analyze_single_upload(uploaded, service).as_outputs()
            except ProofLensError as error:
                message = html.escape(str(error))
                return (
                    None,
                    {},
                    (
                        '<div class="result-card neutral">'
                        "<strong>Cannot analyze image</strong>"
                        f"<span>{message}</span></div>"
                    ),
                    _empty_processing_card(),
                    "Check the upload and try again.",
                    {"error": str(error)},
                )

        analyze.click(
            fn=callback,
            inputs=[image],
            outputs=[
                image_output,
                probabilities,
                verdict,
                processing_card,
                provenance,
                summary,
            ],
        )
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


def _direct_verdict_card(prediction: Prediction, threshold: float) -> str:
    ai_signal = prediction.probability_ai >= threshold
    signal = "AI-generated signal" if ai_signal else "Authentic signal"
    tone = "ai" if ai_signal else "authentic"
    return (
        f'<div class="result-card {tone}">'
        '<span class="eyebrow">Primary result</span>'
        f"<strong>{signal}</strong>"
        f'<span class="result-copy">AI probability {prediction.probability_ai:.1%} · '
        f"Decision threshold {threshold:.1%}</span>"
        f'<span class="pill">{_confidence_label(prediction.confidence)} · '
        f"{prediction.confidence:.1%} confidence</span>"
        "</div>"
    )


def _processing_card(assessment: ProcessingAssessment) -> str:
    if assessment.detected:
        heading = "Possible prior processing"
        labels = ", ".join(html.escape(label) for label in assessment.likely_transformations)
        evidence = " ".join(html.escape(item) for item in assessment.evidence)
        tone = "warning"
    else:
        heading = "No strong processing signal"
        labels = "No visible transformation was identified."
        evidence = html.escape(assessment.evidence[0])
        tone = "stable"
    return (
        f'<div class="result-card {tone}">'
        '<span class="eyebrow">Processing estimate</span>'
        f"<strong>{heading}</strong>"
        f'<span class="result-copy">{labels}</span>'
        f'<span class="result-copy">{evidence}</span>'
        f'<span class="pill">{html.escape(assessment.confidence.title())} confidence · heuristic only</span>'
        f'<span class="result-copy">{html.escape(assessment.caveat)}</span>'
        "</div>"
    )


def _direct_provenance(prediction: Prediction) -> str:
    return (
        f"**Model:** `{prediction.model_version}` · "
        f"**Preprocessing:** `{prediction.preprocessing_version}` · "
        f"**Inference:** `{prediction.inference_ms:.1f} ms`"
    )


def _stability_card(summary: dict[str, object]) -> str:
    _, _, decision = _summary_parts(summary)
    absolute_change = float(summary["absolute_change"])
    condition = html.escape(_transformation_label(str(summary["condition"])))
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


def _transformation_label(condition_id: str) -> str:
    """Return a readable UI label while preserving canonical IDs internally."""

    return _TRANSFORMATION_LABELS.get(condition_id, condition_id.replace("_", " "))


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
        '<span class="result-copy">Upload an image and select Analyze picture.</span></div>'
    )


def _empty_stability_card() -> str:
    return (
        '<div class="result-card neutral"><span class="eyebrow">Robustness</span>'
        "<strong>No comparison yet</strong>"
        '<span class="result-copy">A transformed comparison will appear here.</span></div>'
    )


def _empty_processing_card() -> str:
    return (
        '<div class="result-card neutral"><span class="eyebrow">Processing estimate</span>'
        "<strong>Waiting for an image</strong>"
        '<span class="result-copy">Visible processing signals will appear here.</span></div>'
    )


def _decode_upload(image: Image.Image | None) -> Image.Image:
    if image is None:
        raise UserInputError("Upload an image before analysis")
    try:
        clean_image = image.convert("RGB")
        clean_image.load()
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("Upload a decodable image") from error
    return clean_image


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
