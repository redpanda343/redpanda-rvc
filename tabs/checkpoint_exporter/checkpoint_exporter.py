import os
import sys

import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_checkpoint_export_script

i18n = I18nAuto()

logs_root = os.path.join(now_dir, "logs")


def get_generator_checkpoints():
    if not os.path.isdir(logs_root):
        return []

    return sorted(
        os.path.relpath(os.path.join(root, filename), now_dir)
        for root, _, files in os.walk(logs_root)
        for filename in files
        if filename.startswith("G_") and filename.endswith(".pth")
    )


def refresh_generator_checkpoints():
    return gr.update(choices=get_generator_checkpoints())


def checkpoint_exporter_tab():
    def _export_with_toast(checkpoint_path, precision, output_name):
        gr.Info(i18n("Exporting checkpoint..."))
        try:
            result = run_checkpoint_export_script(
                checkpoint_path, precision, output_name
            )
        except Exception as error:
            gr.Warning(str(error))
            raise
        gr.Info(result[0])
        return result

    gr.Markdown(i18n("## Checkpoint Exporter"))
    gr.Markdown(
        i18n(
            "Convert a resumable G checkpoint into a smaller model for inference."
        )
    )

    generator_checkpoint = gr.Dropdown(
        label=i18n("Generator Checkpoint"),
        info=i18n("Select a saved G checkpoint from your training logs."),
        choices=get_generator_checkpoints(),
        value=None,
        interactive=True,
    )
    refresh_checkpoints = gr.Button(i18n("Refresh Checkpoints"))
    precision = gr.Radio(
        choices=["fp16", "fp32"],
        value="fp16",
        label=i18n("Export Precision"),
        interactive=True,
    )
    output_name = gr.Textbox(
        label=i18n("Output File Name"),
        info=i18n(
            "Choose a name for the exported model. Leave blank to use the automatic name."
        ),
        placeholder=i18n("My exported model"),
        value="",
        max_lines=1,
        interactive=True,
    )
    export_button = gr.Button(i18n("Export Checkpoint"), variant="primary")

    with gr.Row():
        output_info = gr.Textbox(
            label=i18n("Output Information"), interactive=False
        )
        output_model = gr.File(
            label=i18n("Download Model"), type="filepath", interactive=False
        )

    refresh_checkpoints.click(
        fn=refresh_generator_checkpoints,
        inputs=[],
        outputs=[generator_checkpoint],
    )
    export_button.click(
        fn=_export_with_toast,
        inputs=[generator_checkpoint, precision, output_name],
        outputs=[output_info, output_model],
    )
