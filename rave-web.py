import gradio as gr
import os

weight_root = "weights/"
index_root = "logs/"

iw = None

if os.path.exists("infer-web.py"):
    import importlib

    print("Importing infer-web.py ...")
    spec = importlib.util.spec_from_file_location("infer-web.py", "infer-web.py")
    iw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iw)

    print("Done! Loading custom UI")
else:
    print("Not in RVC folder, dry testing UI.")


#
# model name
#
def change_model_and_index_choices():
    names = []
    if os.path.exists(weight_root):
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)
    else:
        names = ["Weights folder not found"]
    index_paths = []

    if os.path.exists(index_root):
        for root, dirs, files in os.walk(index_root, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name:
                    index_paths.append("%s/%s" % (root, name))
    else:
        index_paths = ["Index root folder not found"]
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


#
# def vc_single(
#     sid,
#     input_audio_path,
#     f0_up_key,
#     f0_file,
#     f0_method,
#     file_index,
#     file_index2,
#     # file_big_npy,
#     index_rate,
#     filter_radius,
#     resample_sr,
#     rms_mix_rate,
#     protect,
# ):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0

def process_request(sid, file, transpose, search_feature_ratio, model_index, index_rate, filter_radius, rms_mix_rate,
                    protect):
    print("Received request: " + str(sid) + " " + str(file) + " " + str(transpose) + " " + str(search_feature_ratio))

    f0_method = "crepe"
    file_index1 = ""
    file_index2 = model_index

    if iw is not None:
        return iw.vc_single(sid, file.name, transpose, "", f0_method, file_index1, file_index2, index_rate, filter_radius, 0,
                            rms_mix_rate, protect)
    return "Dry run, not in RVC folder", file.name


def process_model_change(model_name):
    if iw is not None:
        print("Rave-web: Model change to '" + str(model_name) + "'")
        iw.get_vc(model_name, "a", "b")
    else:
        print("Rave-web: Dry model test change to " + str(model_name))
    return "Index of " + str(model_name)


def process_extraction_request(model_choose, file, opt_vocal_root, opt_ins_root, agg, format0):
    print("Extraction request: " + str(model_choose) + " " + str(file.name) + " " + str(opt_vocal_root) + " " + str(opt_ins_root) + " " + str(agg) + " " + str(format0))
    if iw is not None:
        yield iw.uvr(model_choose, "", opt_vocal_root, [file.name], opt_ins_root, agg, format0)
    else:
        return "Dry run " + str(model_choose) + " " + str(file)


def scan_uvr5():
    names = []
    if iw is not None:
        for name in os.listdir(iw.weight_uvr5_root):
            if name.endswith(".pth") or "onnx" in name:
                names.append(name.replace(".pth", ""))
    else:
        names = ["Dry run", "RVC not installed"]
    return names


uvr5_names = scan_uvr5()


with gr.Blocks() as app:
    gr.Markdown(
        """
        # RaveYard's simplified RVC UI
        """
    )
    with gr.Tab("Voice 2 Voice"):

        if iw is None:
            gr.Markdown(
                "ERROR: You must have this file inside RVC folder, because it re-uses some of its code"
            )

        with gr.Row():
            a, b = change_model_and_index_choices()

            with gr.Row(variant='compact'):
                model_list = gr.Dropdown(a['choices'], label="Model", interactive=True)
                model_index = gr.Dropdown(b['choices'], label="Model index", interactive=True)
                model_list.change(process_model_change, [model_list], [model_index])

            with gr.Column():
                with gr.Row():
                    model_refresh_button = gr.Button("Refresh")
                    model_refresh_button.click(change_model_and_index_choices, None, [model_list, model_index])
                    slider_singer_id = gr.Slider(0, 8, value=0, label="Singer ID", step=1)

        with gr.Row():
            with gr.Column():
                input_file = gr.File()
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            slider_transpose = gr.Slider(-36, 36, value=0, label="Transpose (12 is an octave)", step=1)
                    button = gr.Button("Process", variant="primary")

            with gr.Column():
                # TODO what is this?
                slider_sfr = gr.Slider(0.0, 1.0, value=0.5, label="Search feature ratio",
                                       step=0.1)

                # 3 >= uses median filter and reduces breathing.
                slider_filter_radius = gr.Slider(0, 7, value=3, label="Filter radius", step=1)

                # show_label="Higher means stronger accent but more artefacts.
                slider_index_ratio = gr.Slider(0.0, 1.0, value=0.25, label="Accent", step=0.1)

                # low values make the volume consistent with original, high values increase overall volume
                slider_mix_rate = gr.Slider(0.0, 1.0, value=0.25, label="volume envelope scaling", step=0.1)

                # TODO more info?
                slider_protect = gr.Slider(0.0, 1.0, value=0.3, label="Protect non-vocals", step=0.05)

        with gr.Row():
            output_text = gr.Textbox()
            output_file = gr.Audio()

        button.click(process_request,
                     [slider_singer_id, input_file, slider_transpose, slider_sfr, model_index, slider_index_ratio,
                      slider_filter_radius, slider_mix_rate, slider_protect], [output_text, output_file])

        gr.Markdown("""
        ### Note
        This is a limited WEB UI for inference which contains only options which I found relevant.
        If you want to train or extract vocals/instrumental, you'll have to use the OLD ui for now.
        """)

    with gr.Tab("Voice+Instrument Extraction"):
        with gr.Row():
            with gr.Column():
                extract_input_file = gr.File()
                with gr.Row():
                    with gr.Column():
                        model_choose = gr.Dropdown(label="Model", choices=uvr5_names)
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="Vocal extraction aggressiveness",
                            value=10,
                            interactive=True,
                            visible=False,
                        )
                        extract_format = gr.Radio(
                            show_label=False,
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    but2 = gr.Button("Process", variant="primary")
            with gr.Column():
                opt_vocal_root = gr.Textbox(
                    label="Save vocal to folder", value="opt"
                )
                opt_ins_root = gr.Textbox(
                    label="Save instrumental to folder", value="opt"
                )
                vc_output4 = gr.Textbox("", label="Process info")
                with gr.Row():
                    but2.click(
                        iw.single_uvr if iw is not None else process_extraction_request,
                        #process_extraction_request,
                        [
                            model_choose,
                            extract_input_file,
                            opt_vocal_root,
                            opt_ins_root,
                            agg,
                            extract_format,
                        ],
                        [vc_output4]
                    )


if __name__ == "__main__":
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        #inbrowser=not config.noautoopen,
        #server_port=config.listen_port,
        quiet=True
    )
    app.enable_queue = True
    app.launch()
