"""Gradio app over the shared backend-neutral inference service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from prooflens.data.transforms import apply_transform, canonical_specs, get_spec
from prooflens.errors import UserInputError
from prooflens.inference.service import InferenceService


@dataclass(frozen=True, slots=True)
class UploadAnalysis:
    clean_image: Image.Image
    transformed_image: Image.Image
    summary: dict[str, object]

    def as_outputs(
        self,
    ) -> tuple[Image.Image, Image.Image, dict[str, float], dict[str, float], dict[str, object]]:
        clean = self.summary["clean"]
        transformed = self.summary["transformed"]
        if not isinstance(clean, dict) or not isinstance(transformed, dict):
            raise ValueError("analysis summary has invalid prediction values")
        return (
            self.clean_image,
            self.transformed_image,
            {"AI-generated": float(clean["probability_ai"]), "Authentic": float(clean["probability_real"])},
            {"AI-generated": float(transformed["probability_ai"]), "Authentic": float(transformed["probability_real"])},
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
    return UploadAnalysis(
        clean_image=clean_image,
        transformed_image=transformed_image,
        summary={
            "condition": condition_id,
            "clean": asdict(stability.clean),
            "transformed": asdict(stability.transformed),
            "absolute_change": stability.absolute_change,
        },
    )


def create_app(service: InferenceService):
    import gradio as gr

    with gr.Blocks(title="ProofLens") as app:
        gr.Markdown(
            "# ProofLens\n"
            "A research demonstration for authentic-versus-AI-generated image scoring. "
            "Scores are not forensic proof."
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Image")
            condition = gr.Dropdown(
                choices=[spec.condition_id for spec in canonical_specs()],
                value="jpeg_q30", label="Robustness check"
            )
        analyze = gr.Button("Analyze", variant="primary")
        with gr.Row():
            clean_output = gr.Image(label="Clean")
            transformed_output = gr.Image(label="Transformed")
        clean_probabilities = gr.Label(label="Clean probabilities")
        transformed_probabilities = gr.Label(label="Transformed probabilities")
        summary = gr.JSON(label="Prediction stability")
        with gr.Accordion("Model information and limitations", open=False):
            gr.Markdown(
                "The model is sensitive to training-domain and generator shifts. "
                "Compression, resizing, screenshots, and unusual image content can alter scores."
            )

        def callback(uploaded: Image.Image | None, selected: str):
            try:
                return analyze_upload(uploaded, selected, service).as_outputs()
            except UserInputError as error:
                return None, None, {}, {}, {"error": str(error)}

        analyze.click(
            fn=callback,
            inputs=[image, condition],
            outputs=[clean_output, transformed_output, clean_probabilities, transformed_probabilities, summary],
        )
    return app
