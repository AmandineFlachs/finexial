# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### This module contains the chatui gui for having a conversation. ###

import functools
import logging
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

import gradio as gr
import json
import shutil
import os
import subprocess
import time
import torch
import tiktoken
import fnmatch

from chatui import assets, chat_client
from chatui.pages import info
from chatui.pages import utils

_LOGGER = logging.getLogger(__name__)
PATH = "/"
TITLE = "Finexial"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j {
    color: #8F8FD6;
}
"""

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        # create the page header
        gr.Image("chatui/static/Finexial_logo.png", height=75, width=125, min_width=125, show_share_button=False, show_download_button=False, container=False)
    
        # Keep track of state we want to persist across user actions
        which_nim_tab = gr.State(0)
        is_local_nim = gr.State(False)
        vdb_active = gr.State(False)
        metrics_history = gr.State({})
        docs_history = gr.State({})

        # Build the Chat Application
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=15, min_width=350):

                # Main chatbot panel. Context and Metrics are hidden until toggled
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=350):
                        chatbot = gr.Chatbot(show_label=False)
                        
                    context = gr.JSON(
                        scale=1,
                        label="Retrieved Context",
                        visible=False,
                        elem_id="contextbox",
                    )
                    
                    metrics = gr.JSON(
                        scale=1,
                        label="Metrics",
                        visible=False,
                        elem_id="contextbox",
                    )
                    
                    docs = gr.JSON(
                        scale=1,
                        label="Documents",
                        visible=False,
                        elem_id="contextbox",
                    )

                # Render the output sliders to customize the generation output. 
                with gr.Tabs(selected=0, visible=False) as out_tabs:
                    with gr.TabItem("Max Tokens in Response", id=0) as max_tokens_in_response:
                        num_token_slider = gr.Slider(0, utils.preset_max_tokens()[1], value=utils.preset_max_tokens()[0], 
                                                     label=info.num_token_label, 
                                                     interactive=True)
                        
                    with gr.TabItem("Temperature", id=1) as temperature:
                        temp_slider = gr.Slider(0, 1, value=0.7, 
                                                    label=info.temp_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Top P", id=2) as top_p:
                        top_p_slider = gr.Slider(0.001, 0.999, value=0.999, 
                                                    label=info.top_p_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Frequency Penalty", id=3) as freq_pen:
                        freq_pen_slider = gr.Slider(-2, 2, value=0, 
                                                    label=info.freq_pen_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Presence Penalty", id=4) as pres_pen:
                        pres_pen_slider = gr.Slider(-2, 2, value=0, 
                                                    label=info.pres_pen_label, 
                                                    interactive=True)
                        
                    with gr.TabItem("Hide Output Tools", id=5) as hide_out_tools:
                        gr.Markdown("")

                # Hidden button to expand output sliders, if hidden
                out_tabs_show = gr.Button(value="Show Output Tools", size="sm", visible=False)

                # Render the user input textbox and checkbox to toggle vanilla inference and RAG.
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=200):
                        msg = gr.Textbox(
                            show_label=False,
                            lines=3,
                            placeholder="Enter your text and press SUBMIT",
                            container=False,
                            interactive=True,
                        )

                # Render the row of buttons: submit query, clear history, show metrics and contexts
                with gr.Row():
                    submit_btn = gr.Button(value="[NOT READY] Submit", interactive=False)
                    _ = gr.ClearButton([msg, chatbot, metrics, metrics_history], value="Clear History")
                    mtx_show = gr.Button(value="Show Metrics", visible=False)
                    mtx_hide = gr.Button(value="Hide Metrics", visible=False)
                    ctx_hide = gr.Button(value="Hide Context", visible=False)

            # Right Column will display the inference and database settings
            with gr.Column(scale=10, min_width=450, visible=True) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:


                    # This tab item is a button to start the RAG backend and unlock other settings
                    with gr.TabItem("Setup", id=0, interactive=False, visible=True) as setup_settings:
                        gr.Markdown("<br> ")
                        
                        # Select the inference mode settings
                       
                        inference_mode = gr.Radio(["Local System", "Cloud Endpoint"],
                                                  label="Inference Mode",
                                                  info=info.inf_mode_info,
                                                  value="Local System")
                        
                        local_model_id = gr.State("nvidia/Llama3-ChatQA-1.5-8B")
                        local_model_quantize = gr.State("4-Bit")
                        
                        with gr.Row(equal_height=True):
                            download_model = gr.Button(value="Load Model", size="sm", visible=False)
                            start_local_server = gr.Button(value="Start Server", interactive=False, size="sm", visible=False)
                            stop_local_server = gr.Button(value="Stop Server", interactive=False, size="sm", visible=False)

                        nvcf_model_family = gr.State("NVIDIA")
                        nvcf_model_id = gr.State("Llama3 ChatQA-1.5 8B")

                        with gr.Tabs(selected=0, visible=False) as nim_tabs:

                            # Inference settings for remotely-running microservice
                            with gr.TabItem("Remote", id=0) as remote_microservice:
                                remote_nim_msg = gr.Markdown("<br />Enter the details below. Then start chatting!")
                                        
                                with gr.Row(equal_height=True):
                                    nim_model_ip = gr.Textbox(placeholder = "10.123.45.678", 
                                        label = "Microservice Host", 
                                        info = "IP Address running the microservice", 
                                        elem_id="rag-inputs", scale=2)
                                    nim_model_port = gr.Textbox(placeholder = "8000", 
                                        label = "Port", 
                                        info = "Optional, (default: 8000)", 
                                        elem_id="rag-inputs", scale=1)
                                        
                                    nim_model_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct", 
                                        label = "Model running in microservice.", 
                                        info = "If none specified, defaults to: meta/llama3-8b-instruct", 
                                        elem_id="rag-inputs")

                            # Inference settings for locally-running microservice
                            with gr.TabItem("Local", id=1) as local_microservice:
                                gr.Markdown("<br />**Important**: For AI Workbench on DOCKER users only. Podman is unsupported!")
                                        
                                nim_local_model_id = gr.Textbox(placeholder = "nvcr.io/nim/meta/llama3-8b-instruct:latest", 
                                            label = "NIM Container Image", 
                                            elem_id="rag-inputs")
                                        
                                with gr.Row(equal_height=True):
                                    prefetch_nim = gr.Button(value="Prefetch NIM", size="sm")
                                    start_local_nim = gr.Button(value="Start Microservice", 
                                                                interactive=(True if os.path.isdir('/mnt/host-home/model-store') else False), 
                                                                size="sm")
                                    stop_local_nim = gr.Button(value="Stop Microservice", interactive=False, size="sm")
                                                
                        rag_start_button = gr.Button(value="Start Finexial", variant="primary")
                        gr.Markdown("<br> ")

                    # First tab item consists of database and document upload settings
                    with gr.TabItem("Upload Documents Here", id=1, interactive=False, visible=True) as vdb_settings:
                        
                        gr.Markdown(info.update_kb_info)
                        
                        file_output = gr.File(interactive=True, 
                                              show_label=False, 
                                              file_types=["text",
                                                          ".pdf",
                                                          ".html",
                                                          ".doc",
                                                          ".docx",
                                                          ".txt",
                                                          ".odt",
                                                          ".rtf",
                                                          ".tex"], 
                                              file_count="multiple")
        
                        with gr.Row():
                            clear_docs = gr.Button(value="Clear Database", interactive=False, size="sm") 

        def _toggle_hide_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to hide output toolbar from the user. """
            return {
                out_tabs: gr.update(visible=False, selected=0),
                out_tabs_show: gr.update(visible=True),
            }

        hide_out_tools.select(_toggle_hide_out_tools, None, [out_tabs, out_tabs_show])

        def _toggle_show_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to expand output toolbar for the user. """
            return {
                out_tabs: gr.update(visible=True),
                out_tabs_show: gr.update(visible=False),
            }

        out_tabs_show.click(_toggle_show_out_tools, None, [out_tabs, out_tabs_show])

        def _toggle_model_download(btn: str, model: str, start: str, stop: str, inference_mode: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to download model weights locally for Hugging Face TGI local inference. """
            if inference_mode == "Cloud Endpoint":
                return {
                    download_model: gr.update(),
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                }
            elif model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    download_model: gr.update(),
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                }
            else: 
                if btn == "Load Model":
                    progress(0.25, desc="Initializing Task")
                    time.sleep(0.75)
                    progress(0.5, desc="Downloading Model (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/download-local.sh " + model, shell=True)
                    if rc == 0:
                        msg = "Model Downloaded"
                        colors = "primary"
                        interactive = False
                        start_interactive = True if (start == "Start Server") else False
                        stop_interactive = True if (stop == "Stop Server") else False
                    else: 
                        msg = "Error, Try Again"
                        colors = "stop"
                        interactive = True
                        start_interactive = False
                        stop_interactive = False
                progress(0.75, desc="Cleaning Up")
                time.sleep(0.75)
                return {
                    download_model: gr.update(value=msg, variant=colors, interactive=interactive),
                    start_local_server: gr.update(interactive=start_interactive),
                    stop_local_server: gr.update(interactive=stop_interactive),
                }
        
        def _toggle_local_server(btn: str, model: str, quantize: str, download: str, inference_mode: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to run and/or shut down the Hugging Face TGI local inference server. """
            if inference_mode == "Cloud Endpoint":
                return {
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                    msg: gr.update(),
                    submit_btn: gr.update(),
                    download_model: gr.update(),
                }
            elif model != "nvidia/Llama3-ChatQA-1.5-8B" and model != "microsoft/Phi-3-mini-128k-instruct" and model != "" and btn != "Stop Server" and os.environ.get('HUGGING_FACE_HUB_TOKEN') is None:
                gr.Warning("You are accessing a gated model and HUGGING_FACE_HUB_TOKEN is not detected!")
                return {
                    start_local_server: gr.update(),
                    stop_local_server: gr.update(),
                    msg: gr.update(),
                    submit_btn: gr.update(),
                    download_model: gr.update(),
                }
            else: 
                if btn == "Start Server":
                    progress(0.2, desc="Initializing Task")
                    time.sleep(0.5)
                    progress(0.4, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
                    time.sleep(0.5)
                    progress(0.6, desc="Starting Inference Server (may take a few moments)")
                    rc = subprocess.call("/bin/bash /project/code/scripts/start-local.sh " 
                                              + model + " " + utils.quant_to_config(quantize), shell=True)
                    if rc == 0:
                        out = ["Server Started", "Stop Server"]
                        colors = ["primary", "secondary"]
                        interactive = [False, True, True, False]
                    else: 
                        gr.Warning("ERR: You may have timed out or are facing memory issues. In AI Workbench, check Output > Chat for details.")
                        out = ["Internal Server Error, Try Again", "Stop Server"]
                        colors = ["stop", "secondary"]
                        interactive = [False, True, False, False]
                    progress(0.8, desc="Cleaning Up")
                    time.sleep(0.5)
                elif btn == "Stop Server":
                    progress(0.25, desc="Initializing")
                    time.sleep(0.5)
                    progress(0.5, desc="Stopping Server")
                    rc = subprocess.call("/bin/bash /project/code/scripts/stop-local.sh", shell=True)
                    if rc == 0:
                        out = ["Start Server", "Server Stopped"]
                        colors = ["secondary", "primary"]
                        interactive = [True, False, False, False if (download=="Model Downloaded") else True]
                    else: 
                        out = ["Start Server", "Internal Server Error, Try Again"]
                        colors = ["secondary", "stop"]
                        interactive = [True, False, True, False]
                    progress(0.75, desc="Cleaning Up")
                    time.sleep(0.5)
                return {
                    start_local_server: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                    stop_local_server: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                    msg: gr.update(interactive=True, 
                                   placeholder=("Enter text and press SUBMIT" if interactive[2] else "[NOT READY] Start the Local Inference Server OR Select a Different Inference Mode.")),
                    submit_btn: gr.update(value="Submit" if interactive[2] else "[NOT READY] Submit", interactive=interactive[2]),
                    download_model: gr.update(interactive=interactive[3]),
                }

        def _toggle_nim_select(model: str, start: str, stop: str) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to set up user actions for local nim inference. """
            return {
                prefetch_nim: gr.update(value="Prefetch NIM", 
                                               variant="secondary", 
                                               interactive=(False if start == "Microservice Started" else True)),
                start_local_nim: gr.update(interactive=(True if start == "Start Microservice" else False)),
                stop_local_nim: gr.update(interactive=(True if start == "Microservice Started" else False)),
            }
        
        nim_local_model_id.change(_toggle_nim_select,
                              [nim_local_model_id, start_local_nim, stop_local_nim], 
                              [prefetch_nim, start_local_nim, stop_local_nim])

        def _toggle_prefetch_nim(btn: str, model: str, start: str, stop: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to pull the NIM container for local NIM inference. """
            if btn == "Prefetch NIM":
                progress(0.1, desc="Initializing Task")
                list = []
                list.append(model)
                time.sleep(0.25)
                progress(0.3, desc="Checking user configs...")
                if len(model) == 0:
                    gr.Warning("NIM container field cannot be empty. Specify a NIM container to run")
                    msg = "Prefetch NIM"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        prefetch_nim: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                elif len(fnmatch.filter(list, 'nvcr.io/nim/?*/?*')) == 0:
                    gr.Warning("User input is not a valid NIM container image format. Double check the spelling and try again.")
                    msg = "Prefetch NIM"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        prefetch_nim: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/preflight.sh " + model, shell=True)
                if rc != 0:
                    gr.Warning("You may have improper configurations set for this mode. Check the Output > Chat in the AI Workbench UI for details.")
                    msg = "Prefetch NIM"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
                    return {
                        prefetch_nim: gr.update(value=msg, variant=colors, interactive=interactive),
                        start_local_nim: gr.update(interactive=start_interactive),
                        stop_local_nim: gr.update(interactive=stop_interactive),
                    }
                progress(0.6, desc="Pulling NIM container, a one-time process")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/prefetch-nim.sh " + model, shell=True)
                if rc == 0:
                    msg = "Container Pulled"
                    colors = "primary"
                    interactive = False
                    start_interactive = True if (start == "Start Microservice") else False
                    stop_interactive = True if (stop == "Stop Microservice") else False
                else: 
                    gr.Warning("Ran into an error pulling the NIM container. Is your NGC_CLI_API_KEY correct? Check the Output > Chat in the AI Workbench UI for details.")
                    msg = "Prefetch NIM"
                    colors = "secondary"
                    interactive = True
                    start_interactive = False
                    stop_interactive = False
            progress(0.9, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                prefetch_nim: gr.update(value=msg, variant=colors, interactive=interactive),
                start_local_nim: gr.update(interactive=start_interactive),
                stop_local_nim: gr.update(interactive=stop_interactive),
            }
        
        prefetch_nim.click(_toggle_prefetch_nim,
                             [prefetch_nim, nim_local_model_id, start_local_nim, stop_local_nim], 
                             [prefetch_nim, start_local_nim, stop_local_nim, msg])
        
        def _toggle_local_nim(btn: str, model: str, prefetched_nim: str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener for running and/or shutting down the local nim sidecar container. """
            if btn == "Start Microservice":
                progress(0.2, desc="Initializing Task")
                list = []
                list.append(model)
                time.sleep(0.25)
                progress(0.4, desc="Checking user configs...")
                if len(fnmatch.filter(list, 'nvcr.io/nim/?*/?*')) == 0:
                    gr.Warning("User input is not a valid NIM container image format. Double check the spelling and try again.")
                    out = ["Start Microservice", "Stop Microservice"]
                    colors = ["secondary", "secondary"]
                    interactive = [False, True, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value=""
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    prefetch_nim_interactive = True
                    return {
                        start_local_nim: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                        stop_local_nim: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                        nim_local_model_id: gr.update(interactive=interactive[2]),
                        remote_nim_msg: gr.update(value=value),
                        which_nim_tab: submittable, 
                        submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                        prefetch_nim: gr.update(interactive=prefetch_nim_interactive),
                        msg: gr.update(interactive=True, 
                                       placeholder=("Enter text and press SUBMIT" if interactive[3] else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode.")),
                    }
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/preflight.sh " + model, shell=True)
                if rc != 0:
                    gr.Warning("You may have improper configurations set for this mode. Check the Output > Chat in the AI Workbench UI for details.")
                    out = ["Internal Server Error, Try Again", "Stop Microservice"]
                    colors = ["stop", "secondary"]
                    interactive = [False, True, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value=""
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    prefetch_nim_interactive = True
                    return {
                        start_local_nim: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                        stop_local_nim: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                        nim_local_model_id: gr.update(interactive=interactive[2]),
                        remote_nim_msg: gr.update(value=value),
                        which_nim_tab: submittable, 
                        submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                        prefetch_nim: gr.update(interactive=prefetch_nim_interactive),
                        msg: gr.update(interactive=True, 
                                       placeholder=("Enter text and press SUBMIT" if interactive[3] else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode.")),
                    }
                progress(0.6, desc="Starting Microservice, may take a moment")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/start-local-nim.sh " + model + " " + utils.nim_extract_model(model), shell=True)
                if rc == 0:
                    out = ["Microservice Started", "Stop Microservice"]
                    colors = ["primary", "secondary"]
                    interactive = [False, True, False, True]
                    model_ip = ["local_nim"]
                    model_id = [model]
                    value="<br />Stop the local microservice before using a remote microservice."
                    submit_value = "Submit"
                    submittable = 0
                    prefetch_nim_interactive = False
                else: 
                    gr.Warning("Ran into an issue starting up the NIM Container. Double check the spelling, and see Troubleshooting for details. ")
                    out = ["Internal Server Error, Try Again", "Stop Microservice"]
                    colors = ["stop", "secondary"]
                    interactive = [False, True, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value=""
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    prefetch_nim_interactive = True if prefetched_nim == "Prefetch NIM" else False
                progress(0.8, desc="Cleaning Up")
                time.sleep(0.5)
            elif btn == "Stop Microservice":
                progress(0.25, desc="Initializing")
                time.sleep(0.5)
                progress(0.5, desc="Stopping Microservice")
                rc = subprocess.call("/bin/bash /project/code/scripts/local-nim-configs/stop-local-nim.sh ", shell=True)
                if rc == 0:
                    out = ["Start Microservice", "Microservice Stopped"]
                    colors = ["secondary", "primary"]
                    interactive = [True, False, True, False]
                    model_ip = [""]
                    model_id = [""]
                    value="<br />Enter the details below. Then start chatting!"
                    submit_value = "[NOT READY] Submit"
                    submittable = 1
                    prefetch_nim_interactive = True if prefetched_nim == "Prefetch NIM" else False
                else: 
                    gr.Warning("Ran into an issue stopping the NIM Container, try again. The service may still be running. ")
                    out = ["Start Microservice", "Internal Server Error, Try Again"]
                    colors = ["secondary", "stop"]
                    interactive = [True, False, True, True]
                    model_ip = ["local_nim"]
                    model_id = [model]
                    value=""
                    submit_value = "Submit"
                    submittable = 0
                    prefetch_nim_interactive = False
                progress(0.75, desc="Cleaning Up")
                time.sleep(0.5)
            return {
                start_local_nim: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                stop_local_nim: gr.update(value=out[1], variant=colors[1], interactive=interactive[1]),
                nim_local_model_id: gr.update(interactive=interactive[2]),
                remote_nim_msg: gr.update(value=value),
                which_nim_tab: submittable, 
                submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                prefetch_nim: gr.update(interactive=prefetch_nim_interactive),
                msg: gr.update(interactive=True, 
                               placeholder=("Enter text and press SUBMIT" if interactive[3] else "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode.")),
            }

        start_local_nim.click(_toggle_local_nim, 
                                 [start_local_nim, 
                                  nim_local_model_id,
                                  prefetch_nim], 
                                 [start_local_nim, 
                                  stop_local_nim, 
                                  nim_local_model_id, 
                                  remote_nim_msg,
                                  which_nim_tab, 
                                  submit_btn,
                                  prefetch_nim,
                                  msg])
        stop_local_nim.click(_toggle_local_nim, 
                                 [stop_local_nim, 
                                  nim_local_model_id,
                                  prefetch_nim], 
                                 [start_local_nim, 
                                  stop_local_nim, 
                                  nim_local_model_id, 
                                  remote_nim_msg,
                                  which_nim_tab, 
                                  submit_btn,
                                  prefetch_nim,
                                  msg])

        def _toggle_kb(btn: str, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to clear the vector database of all documents. """
            if btn == "Clear Database":
                progress(0.25, desc="Initializing Task")
                update_docs_uploaded = docs_uploaded
                time.sleep(0.25)
                progress(0.5, desc="Clearing Vector Database")
                success = utils.clear_knowledge_base()
                if success:
                    out = ["Clear Database"]
                    colors = ["secondary"]
                    interactive = [True]
                    progress(0.75, desc="Success!")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Deleted"})
                    time.sleep(0.5)
                else: 
                    gr.Warning("Your files may still be present in the database. Try again.")
                    out = ["Error Clearing Vector Database"]
                    colors = ["stop"]
                    interactive = [True]
                    progress(0.75, desc="Error, try again.")
                    for key, value in update_docs_uploaded.items():
                        update_docs_uploaded.update({str(key): "Unknown"})
                    time.sleep(0.5)
            else: 
                out = ["Clear Database"]
                colors = ["secondary"]
                interactive = [True]
            return {
                file_output: gr.update(value=None, 
                                       interactive=True, 
                                       show_label=False, 
                                       file_types=["text",
                                                   ".pdf",
                                                   ".html",
                                                   ".doc",
                                                   ".docx",
                                                   ".txt",
                                                   ".odt",
                                                   ".rtf",
                                                   ".tex"], 
                                       file_count="multiple"),
                clear_docs: gr.update(value=out[0], variant=colors[0], interactive=interactive[0]),
                #kb_checkbox: gr.update(value=None),
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
            }
            
        clear_docs.click(_toggle_kb, [clear_docs, docs_history], [clear_docs, file_output, msg, docs, docs_history])

        def _document_upload(files, docs_uploaded, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to upload documents to the vector database. """
            progress(0.25, desc="Initializing Task")
            update_docs_uploaded = docs_uploaded
            time.sleep(0.25)
            progress(0.5, desc="Polling Vector DB Status")
            rc = subprocess.call("/bin/bash /project/code/scripts/check-database.sh ", shell=True)
            if rc == 0:
                progress(0.75, desc="Pushing uploaded files to DB...")
                file_paths = utils.upload_file(files, client)
                success=True
                for file in file_paths:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Uploaded Successfully"})
            else: 
                gr.Warning("Hang Tight! The Vector DB may be temporarily busy. Give it a moment, and then try again. ")
                file_paths = None
                success=False
                file_names = [file.name for file in files]
                for file in file_names:
                    update_docs_uploaded.update({str(file.split('/')[-1]): "Failed to Upload"})
            return {
                file_output: gr.update(value=file_paths),
                docs: gr.update(value=update_docs_uploaded),
                docs_history: update_docs_uploaded,
                clear_docs: gr.update(interactive=True)
            }

        file_output.upload(_document_upload, [file_output, docs_history], [file_output, docs, docs_history, clear_docs])

        def _toggle_rag_start(btn: str, inference_mode:str, progress=gr.Progress()) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to initialize the RAG backend API server and start warming up the vector database. """
            progress(0.25, desc="Initializing Task")
            time.sleep(0.25)
            progress(0.5, desc="Setting Up RAG Backend (one-time process, may take a few moments)")
            rc = subprocess.call("/bin/bash /project/code/scripts/rag-consolidated.sh ", shell=True)
            if rc == 2:
                gr.Info("Inferencing is ready, but the Vector DB may still be spinning up. This can take a few moments to complete. ")
                visibility = [False, True, True, True]
                interactive = [False, True, True, False] if inference_mode == "Local System" else [False, True, True, True]
                submit_value="[NOT READY] Submit" if inference_mode == "Local System" else "Submit"
            elif rc == 0:
                visibility = [False, True, True, True]
                interactive = [False, True, True, False]
                submit_value="[NOT READY] Submit"
            else:
                gr.Warning("Something went wrong. Check the Output in AI Workbench, or try again. ")
                visibility = [True, True, True, False]
                interactive = [False, False, False, False]
                submit_value="[NOT READY] Submit"
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.25)
            return {
                setup_settings: gr.update(visible=visibility[0], interactive=interactive[0]), 
                vdb_settings: gr.update(visible=visibility[2], interactive=interactive[2]),
                submit_btn: gr.update(value=submit_value, interactive=interactive[3]),
                msg: gr.update(interactive=True, placeholder="[NOT READY] Select a model OR Select a Different Inference Mode." if rc != 1 and inference_mode == "Local System" else "Enter text and press SUBMIT"),
            }

        rag_start_button.click(_toggle_rag_start, [rag_start_button, inference_mode], [setup_settings, vdb_settings, submit_btn, msg]).then(_toggle_model_download,
            [download_model, local_model_id, start_local_server, stop_local_server, inference_mode],
            [download_model, start_local_server, stop_local_server, msg]).then(_toggle_local_server,
            [start_local_server, local_model_id, local_model_quantize, download_model, inference_mode], 
            [start_local_server, stop_local_server, msg, submit_btn, download_model])

        def _toggle_remote_ms() -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select the remote-microservice inference mode for microservice inference. """
            return {
                which_nim_tab: 0, 
                is_local_nim: False, 
                submit_btn: gr.update(value="Submit", interactive=True),
                msg: gr.update(placeholder="Enter text and press SUBMIT")
            }
        
        remote_microservice.select(_toggle_remote_ms, None, [which_nim_tab, is_local_nim, submit_btn, msg])

        def _toggle_local_ms(start_btn: str, stop_btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            """ Event listener to select the local-nim inference mode for microservice inference. """
            if (start_btn == "Microservice Started"):
                interactive = True
                submit_value = "Submit"
                msg_value = "Enter text and press SUBMIT"
                submittable = 0
            elif (start_btn == "Start Microservice" and stop_btn == "Stop Microservice"): 
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."
                submittable = 1
            else:
                interactive = False
                submit_value = "[NOT READY] Submit"
                msg_value = "[NOT READY] Start the Local Microservice OR Select a Different Inference Mode."
                submittable = 1
            return {
                which_nim_tab: submittable, 
                is_local_nim: True, 
                submit_btn: gr.update(value=submit_value, interactive=interactive),
                msg: gr.update(placeholder=msg_value)
            }
        
        local_microservice.select(_toggle_local_ms, [start_local_nim, stop_local_nim], [which_nim_tab, is_local_nim, submit_btn, msg])
        
        # form actions
        _my_build_stream = functools.partial(_stream_predict, client)
        msg.submit(
            _my_build_stream, [inference_mode, 
                               nvcf_model_id, 
                               nim_model_ip, 
                               nim_model_port, 
                               nim_local_model_id,
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               top_p_slider, 
                               freq_pen_slider, 
                               pres_pen_slider,
                               start_local_server,
                               local_model_id,
                               msg, 
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history]
        )
        submit_btn.click(
            _my_build_stream, [inference_mode, 
                               nvcf_model_id, 
                               nim_model_ip, 
                               nim_model_port, 
                               nim_local_model_id,
                               nim_model_id,
                               is_local_nim, 
                               num_token_slider,
                               temp_slider,
                               top_p_slider, 
                               freq_pen_slider, 
                               pres_pen_slider,
                               start_local_server,
                               local_model_id,
                               msg, 
                               metrics_history,
                               chatbot], [msg, chatbot, context, metrics, metrics_history]
        )

    page.queue()
    return page

def _stream_predict(
    client: chat_client.ChatClient,
    inference_mode: str,
    nvcf_model_id: str,
    nim_model_ip: str,
    nim_model_port: str,
    nim_local_model_id: str, 
    nim_model_id: str,
    is_local_nim: bool,
    num_token_slider: float, 
    temp_slider: float, 
    top_p_slider: float, 
    freq_pen_slider: float, 
    pres_pen_slider: float, 
    start_local_server: str,
    local_model_id: str,
    question: str,
    metrics_history: dict,
    chat_history: List[Tuple[str, str]],
) -> Any:
    """
    Make a prediction of the response to the prompt.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
        inference_mode (str): The inference mode selected for this query
        nvcf_model_id (str): The cloud endpoint selected for this query
        nim_model_ip (str): The ip address running the remote nim selected for this query
        nim_model_port (str): The port for the remote nim selected for this query
        nim_local_model_id (str): The model name for local nim selected for this query
        nim_model_id (str): The model name for remote nim selected for this query
        is_local_nim (bool): Whether to run the query as local or remote nim
        num_token_slider (float): max number of tokens to generate
        temp_slider (float): temperature selected for this query
        top_p_slider (float): top_p selected for this query
        freq_pen_slider (float): frequency penalty selected for this query 
        pres_pen_slider (float): presence penalty selected for this query
        start_local_server (str): local TGI server status
        local_model_id (str): model name selected for local TGI inference of this query
        question (str): user prompt
        metrics_history (dict): current list of generated metrics
        chat_history (List[Tuple[str, str]]): current history of chatbot messages
    
    Returns:
        (Dict[gr.component, Dict[Any, Any]]): Gradio components to update.
    """

    chunks = ""

    # Input validation for remote microservice settings
    if (utils.inference_to_config(inference_mode) == "microservice" and
        (len(nim_model_ip) == 0) and 
        is_local_nim == False):
        yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: Hostname/IP field cannot be empty. "]], None, gr.update(value=metrics_history), metrics_history

    # Inputs are validated, can proceed with generating a response to the user query.
    else:

        # Try to send a request for the query
        try:
            documents: Union[None, List[Dict[str, Union[str, float]]]] = None
            response_num = len(metrics_history.keys())
            retrieval_ftime = ""
            chunks = ""
            e2e_stime = time.time()
            retrieval_stime = time.time()
            documents = client.search(question)
            retrieval_ftime = str((time.time() - retrieval_stime) * 1000).split('.', 1)[0]

            # Generate the output
            chunk_num = 0
            for chunk in client.predict(question, 
                                        utils.inference_to_config(inference_mode), 
                                        local_model_id,
                                        utils.cloud_to_config(nvcf_model_id), 
                                        "local_nim" if is_local_nim else nim_model_ip, 
                                        "8000" if is_local_nim else nim_model_port, 
                                        utils.nim_extract_model(nim_local_model_id) if is_local_nim else nim_model_id,
                                        temp_slider,
                                        top_p_slider,
                                        freq_pen_slider,
                                        pres_pen_slider,
                                        True, 
                                        int(num_token_slider)):
                
                # The first chunk returned will always be the time to first token. Let's process that first.
                if chunk_num == 0:
                    chunk_num += 1
                    ttft = chunk
                    updated_metrics_history = utils.get_initial_metrics(metrics_history, response_num, inference_mode, nvcf_model_id, local_model_id, 
                                                                        nim_local_model_id, is_local_nim, nim_model_id, retrieval_ftime, ttft)
                    yield "", chat_history, documents, gr.update(value=updated_metrics_history), updated_metrics_history
                
                # Every next chunk will be the generated response. Let's append to the output and render it in real time. 
                else:
                    chunks += chunk
                    chunk_num += 1
                yield "", chat_history + [[question, chunks]], documents, gr.update(value=metrics_history), metrics_history

            # With final output generated, run some final calculations and display them as metrics to the user
            gen_time, e2e_ftime, tokens, tokens_sec, itl = utils.get_final_metrics(time.time(), e2e_stime, ttft, retrieval_ftime, chunks)
            metrics_history.get(str(response_num)).update({"Generation Time": gen_time + "ms", 
                                                           "End to End Time (E2E)": e2e_ftime + "ms", 
                                                           "Tokens (est.)": tokens + " tokens", 
                                                           "Tokens/Second (est.)": tokens_sec + " tokens/sec", 
                                                           "Inter-Token Latency (est.)": itl + " ms"})
            yield "", gr.update(show_label=False), documents, gr.update(value=metrics_history), metrics_history

        # Catch any exceptions and direct the user to the logs/output. 
        except Exception as e: 
            yield "", chat_history + [[question, "*** ERR: Unable to process query. ***\n\nMessage: " + str(e)]], None, gr.update(value=metrics_history), metrics_history
