import os
import sys

import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_model_blender_script

i18n = I18nAuto()

logs_root = os.path.join(now_dir, "logs")


def get_blendable_models():
    if not os.path.isdir(logs_root):
        return []

    return sorted(
        os.path.relpath(os.path.join(root, filename), now_dir)
        for root, _, files in os.walk(logs_root)
        for filename in files
        if filename.endswith(".pth")
        and not filename.startswith(("G_", "D_"))
    )


def select_uploaded_model(model_path):
    if not model_path:
        return gr.update(), gr.update()
    gr.Info(i18n("Uploaded model selected."))
    return gr.update(value=model_path), None


def refresh_logged_models():
    choices = get_blendable_models()
    return gr.update(choices=choices), gr.update(choices=choices)


def voice_blender_tab():
    def _blend_with_toast(model_name, model_a, model_b, blend_position):
        gr.Info(i18n("Blending models..."))
        try:
            model_a_ratio = 1 - blend_position
            result = run_model_blender_script(
                model_name, model_a, model_b, model_a_ratio
            )
        except Exception:
            gr.Warning(
                i18n(
                    "An error occurred blending the models. Please check the console logs for more details."
                )
            )
            raise
        message = result[0] if isinstance(result, tuple) else result
        if isinstance(message, str):
            if "error" in message.lower() or "failed" in message.lower():
                gr.Warning(message)
            else:
                gr.Info(message)
        return result

    gr.Markdown(i18n("## Voice Blender"))
    gr.Markdown(
        i18n(
            "Select two voice models, set your desired blend percentage, and blend them into an entirely new voice."
        )
    )
    with gr.Column():
        model_fusion_name = gr.Textbox(
            label=i18n("Model Name"),
            info=i18n("Name of the new model."),
            value="",
            max_lines=1,
            interactive=True,
            placeholder=i18n("Enter model name"),
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(i18n("### Model A"))
                model_fusion_a_dropdown = gr.Dropdown(
                    label=i18n("Select a Model"),
                    info=i18n("Choose a model located in your logs folder."),
                    choices=get_blendable_models(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
                model_fusion_a_upload = gr.File(
                    label=i18n("Or Upload Model A"),
                    file_types=[".pth"],
                    type="filepath",
                )
            with gr.Column():
                gr.Markdown(i18n("### Model B"))
                model_fusion_b_dropdown = gr.Dropdown(
                    label=i18n("Select a Model"),
                    info=i18n("Choose a model located in your logs folder."),
                    choices=get_blendable_models(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
                model_fusion_b_upload = gr.File(
                    label=i18n("Or Upload Model B"),
                    file_types=[".pth"],
                    type="filepath",
                )
        refresh_models = gr.Button(i18n("Refresh Models"))
        gr.Markdown(i18n("**Model A ← Blend → Model B**"))
        blend_position = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.01,
            label=i18n("Blend Ratio"),
            value=0.5,
            interactive=True,
            info=i18n(
                "Move left for more Model A or right for more Model B. The center blends both equally."
            ),
        )
        model_fusion_button = gr.Button(i18n("Fusion"))
        with gr.Row():
            model_fusion_output_info = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
                value="",
            )
            model_fusion_pth_output = gr.File(
                label=i18n("Download Model"), type="filepath", interactive=False
            )

    model_fusion_button.click(
        fn=_blend_with_toast,
        inputs=[
            model_fusion_name,
            model_fusion_a_dropdown,
            model_fusion_b_dropdown,
            blend_position,
        ],
        outputs=[model_fusion_output_info, model_fusion_pth_output],
    )

    model_fusion_a_upload.upload(
        fn=select_uploaded_model,
        inputs=[model_fusion_a_upload],
        outputs=[model_fusion_a_dropdown, model_fusion_a_upload],
    )
    model_fusion_b_upload.upload(
        fn=select_uploaded_model,
        inputs=[model_fusion_b_upload],
        outputs=[model_fusion_b_dropdown, model_fusion_b_upload],
    )
    refresh_models.click(
        fn=refresh_logged_models,
        inputs=[],
        outputs=[model_fusion_a_dropdown, model_fusion_b_dropdown],
    )
