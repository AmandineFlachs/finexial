# Finexial - Project summary [![Open In AI Workbench](https://img.shields.io/badge/Open_In-AI_Workbench-76B900)](https://github.com/AmandineFlachs/finexial.git)

<!-- Banner Image 
<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/rag-representation.jpg" width="100%">-->

<!-- Links
<p align="center"> 
  <a href="https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/" style="color: #76B900;">:arrow_down: Download AI Workbench</a> •
  <a href="https://docs.nvidia.com/ai-workbench/" style="color: #76B900;">:book: Read the Docs</a> •
  <a href="https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html" style="color: #76B900;">:open_file_folder: Explore Example Projects</a> •
  <a href="https://forums.developer.nvidia.com/t/support-workbench-example-project-agentic-rag/303414" style="color: #76B900;">:rotating_light: Facing Issues? Let Us Know!</a>
</p> -->

Finexial is an AI RAG project created for the [HackAI - Dell and NVIDIA Challenge](https://hackaichallenge.devpost.com). 
It aims to help non-financial employees make the most of their financial data and feel more confident working with large financial reports.

Companies expect 70% of their employee to heavily use data to make decisions by 2025, a jump of over 40% since 2018 (Tableau, 2021). Still, 74% of employees report feeling overwhelmed when working with data (Accenture, 2020). While a majority of business decisions are influenced by financial data, only 30% of employees from non-financial departments feel confident in their ability to understand and use financial information effectively (Deloitte, 2021). In the USA only, the lack of data and financial literacy is estimated to cost companies over $100B per year (Accenture, 2020).

**Finexial is an AI-powered tool designed to understand financial reports, discuss reports' outcome and explain any financial jargon.**

## Repository

The following files are included in the repository
IMAGE TO ADD

## Set up and run Finexial

Finexial has been designed to be easy to run and use thanks to [Nvidia AI Workbench](https://www.nvidia.com/en-gb/deep-learning-ai/solutions/data-science/workbench/). It currently relies on the non-gated [nvidia/Llama3-ChatQA-1.5-8B](https://build.nvidia.com/nvidia/chatqa-1-5-8b) model. 

### Follow the steps to set up Finxial:
* Install and configure [Nvidia AI Workbench](https://www.nvidia.com/en-gb/deep-learning-ai/solutions/data-science/workbench/) locally and open up AI Workbench [(see the documentation for more info)](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html). NVIDIA AI Workbench is a free client application that handles Git repositories and containers and manages dev environments. 
* Fork this repository into your own GitHub account.
* Inside AI Workbench:
    * Click Clone Project and enter the GitHub repository URL of your newly-forked repo.
    * AI Workbench will automatically clone the repo and build out the project environment, which can take a while to complete.
    * In the Environment tab, add your own token (key) in the Secret section for both the HUGGING_FACE_HUB_TOKEN and NVCF_RUN_KEY. You will need to create a free account respectively on [Hugging Face](https://huggingface.co) and [NVIDIA NGN](https://ngc.nvidia.com/signin) even if you plan to use Finexial locally.
    * Still in the Environment tab, click on Start Environment.
    * Then click on Start Chat to launch the application. A new tab in your web browser will automatically open. 

You are now ready to go!

### How to use Finexial

Now that the system is set up, you can run Finexial:
1. Choose your inference more (local or cloud).
2. Click on the button Start Finexial.
3. Upload your document(s) in pdf.
4. Use the chat box to ask questions about the reports and documents uploaded using natural language. At anytime you can start again from scratch by clearing your chat history (Clear History button) or remote the document(s) uploaded (Clear Database button). 

### Hardware requirements for local inference

While there are no hardware requirements for cloud inference, if you are running Finexial locally you will need to ensure you have sufficient GPU resources. For this project, I used an Nvidia RTX 3090. 

## How it works
    
### Table 1 Default Supported Models by Inference Mode

<details>
<summary>
<b>Expand this section for a full table on all supported models by inference mode.</b>
</summary>

 | Model    | Local ([TGI](https://huggingface.co/docs/text-generation-inference/en/index))         | Cloud ([NVIDIA API Catalog](https://build.nvidia.com/explore/discover)) | Microservices ([NVIDIA NIMs](https://www.nvidia.com/en-us/ai/)) |
 | -------- | ----------------------------- | --------------- | ------------------------------ |
 | Llama3-ChatQA-1.5-8B |           Y       |     Y           | *                              |
 | Llama3-ChatQA-1.5-70B |                  |     Y           | *                              |
 | Nemotron-Mini-4B |                       |     Y           | *                              |
 | Nemotron-4-340B-Instruct |               |     Y           | *                              |
 | Mistral-NeMo 12B Instruct |              |     Y           | *                              |
 | Mistral-7B-Instruct-v0.1 |    Y (gated)  |                 | *                              |
 | Mistral-7B-Instruct-v0.2 |    Y (gated)  |     Y           | *                              |
 | Mistral-7B-Instruct-v0.3 |               |     Y           | Y                              |
 | Mistral-Large |                          |     Y           | *                              |
 | Mixtral-8x7B-Instruct-v0.1 |             |     Y           | Y                              |
 | Mixtral-8x22B-Instruct-v0.1 |            |     Y           | Y                              |
 | Mamba Codestral 7B v0.1 |                |     Y           | *                              |
 | Codestral-22B-Instruct-v0.1 |            |     Y           | *                              |
 | Llama-2-7B-Chat |             Y (gated)  |                 | *                              |
 | Llama-2-13B-Chat |                       |                 | *                              |
 | Llama-2-70B-Chat |                       |     Y           | *                              |
 | Llama-3-8B-Instruct |         Y (gated)  |     Y           | Y (default) *                  |
 | Llama-3-70B-Instruct |                   |     Y           | Y                              |
 | Llama-3.1-8B-Instruct |                  |     Y           | *                              |
 | Llama-3.1-70B-Instruct |                 |     Y           | *                              |
 | Llama-3.1-405B-Instruct |                |     Y           | *                              |
 | Gemma-2B |                               |     Y           | *                              |
 | Gemma-7B |                               |     Y           | *                              |
 | CodeGemma-7B |                           |     Y           | *                              |
 | Phi-3-Mini-4k-Instruct |                 |     Y           | *                              |
 | Phi-3-Mini-128k-Instruct |    Y          |     Y           | *                              |
 | Phi-3-Small-8k-Instruct |                |     Y           | *                              |
 | Phi-3-Small-128k-Instruct |              |     Y           | *                              |
 | Phi-3-Medium-4k-Instruct |               |     Y           | *                              |
 | Phi-3-Medium-128k-Instruct |             |     Y           | *                              |
 | Phi-3.5-Mini-Instruct |                  |     Y           | *                              |
 | Phi-3.5-MoE-Instruct |                   |     Y           | *                              |
 | Arctic |                                 |     Y           | *                              |
 | Granite-8B-Code-Instruct |               |     Y           | *                              |
 | Granite-34B-Code-Instruct |              |     Y           | *                              |
 | Solar-10.7B-Instruct |                   |     Y           | *                              |
 | Jamba-1.5-Mini-Instruct |                |     Y           | *                              |
 | Jamba-1.5-Large-Instruct |               |     Y           | *                              |

</details>

*NIM containers for LLMs are starting to roll out under [General Availability (GA)](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama3-8b-instruct/tags). If you set up any accessible language model NIM running on another system, it is supported under Remote NIM inference inside this project. For Local NIM inference, this project provides a flow for setting up the default ``meta/llama3-8b-instruct`` NIM locally as an example. Advanced users may choose to swap this NIM Container Image out with other NIMs as they are released. 

# Quickstart
This section demonstrates how to use this project to run RAG via **NVIDIA Inference Endpoints** hosted on the [NVIDIA API Catalog](https://build.nvidia.com/explore/discover). For other inference options, including local inference, see the [Advanced Tutorials](#advanced-tutorials) section for set up and instructions.

## Prerequisites
- An [NGC account](https://ngc.nvidia.com/signin) is required to generate an NVCF run key. 
- A valid NVCF key is required to access NVIDIA API endpoints. Generate a key on any NVIDIA API catalog model card, eg. [here](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2) by clicking "Get API Key". 

## Tutorial: Using a Cloud Endpoint

<img src="./code/chatui/static/cloud.gif" width="85%" height="auto">

1. [Install and configure](#nvidia-ai-workbench) AI Workbench locally and open up AI Workbench. Select a location of your choice. 
2. Fork this repo into *your own* GitHub account.
3. **Inside AI Workbench**:
    - Click **Clone Project** and enter the repo URL of your newly-forked repo.
    - AI Workbench will automatically clone the repo and build out the project environment, which can take several minutes to complete. 
    - Upon `Build Complete`, navigate to **Environment** > **Secrets** > **NVCF_RUN_KEY** > **Configure** and paste in your NVCF run key as a project secret. 
    - Select **Open Chat** on the top right of the AI Workbench window, and the Gradio app will open in a browser. 
4. **In the Gradio Chat app**:
    - Click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
    - Select the **Cloud** option, select a model family and model name, and submit a query. 
    - To perform RAG, select **Upload Documents Here** from the right hand panel of the chat UI.
         - You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
    - When the database starts, select **Click to Upload** and choose the text files to upload.
    - Once the files upload, the **Toggle to Use Vector Database** next to the text input box will turn on.
    - Now query your documents! What are they telling you?
    - To change the endpoint, choose a different model from the dropdown on the right-hand settings panel and continue querying.

---
**Next Steps:**
* If you get stuck, check out the ["Troubleshooting"](#troubleshooting) section.
* For tutorials on other supported inference modes, check out the ["Advanced Tutorials"](#advanced-tutorials) section below. **Note:** All subsequent tutorials will assume ``NVCF_RUN_KEY`` is already configured with your credentials. 

---

# Troubleshooting

Need help? Submit any questions, bugs, feature requests, and feedback at the Developer Forum for AI Workbench. The dedicated thread for this Hybrid RAG example project is located [here](https://forums.developer.nvidia.com/t/support-workbench-example-project-hybrid-rag/288565). 

### How do I open AI Workbench?
- Make sure you [installed](#nvidia-ai-workbench) AI Workbench. There should be a desktop icon on your system. Double click it to start AI Workbench.

    <img src="./code/chatui/static/desktop-app.png" width="10%" height="auto">

### How do I clone this repo with AI Workbench?
- Make sure you have opened AI Workbench.
- Click on the **Local** location (or whatever location you want to clone into).
- If this is your first project, click the green **Clone Existing Project** button.
    - Otherwise, click **Clone Project** in the top right
- Drop in the repo URL, leave the default path, and click **Clone**. 

    <img src="./code/chatui/static/clone.png" width="66%" height="auto">

### I've cloned the project, but now nothing seems to be happening?
- The container is likely building and can take several minutes.
- Look at the very <ins>bottom</ins> of the Workbench window, you will see a **Build Status** widget.
- Click it to expand the build output. 
- When the container is built, the widget will say `Build Ready`.
- Now you can begin. 

    <img src="./code/chatui/static/built.png" width="66%" height="auto">

### How do I start the Chat application?
- Check that the container finished building.
- When it finishes, click the green **Open Chat** button at the top right.

    <img src="./code/chatui/static/chat.png" width="66%" height="auto">

### Something went wrong, how do I debug the Chat application?
- Look at the bottom left of the AI Workbench window, you will see an **Output** widget.
- Click it to expand the output. 
- Expand the dropdown, navigate to ``Applications`` > ``Chat``.
- You can now view all debug messages in the Output window in real time.

    <img src="./code/chatui/static/debug-chat.png" width="66%" height="auto">

### How can I customize this project with AI Workbench?
- Check that the container is built.
- Then click the green **dropdown** next to the `Open Chat` button at the top right.
- Select **JupyterLab** to start editing the code. Alternatively, you may configure VSCode support [here](https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications/built-in/vs-code.html).

    <img src="./code/chatui/static/jupyter.png" width="66%" height="auto">

# Advanced Tutorials
This section shows you how to use different inference modes with this RAG project. For these tutorials, a GPU of at least 12 GB of vRAM is recommended. If you don't have one, go back to the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint) that shows how to use **Cloud Endpoints**. 

## Tutorial 1: Using a local GPU
This tutorial assumes you already cloned this Hybrid RAG project to your AI Workbench. If not, please follow the beginning of the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint). 

<img src="./code/chatui/static/local.gif" width="85%" height="auto">

### Additional Configurations

#### Ungated Models
The following models are _ungated_. These can be accessed, downloaded, and run locally inside the project with no additional configurations required:
  * [nvidia/Llama3-ChatQA-1.5-8B](https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B)
  * [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

#### Gated models
Some additional configurations in AI Workbench are required to run certain listed models. Unlike the previous tutorials, these configs are not added to the project by default, so please follow the following instructions closely to ensure a proper setup. Namely, a Hugging Face API token is required for running gated models locally. [See how to create a token here](https://huggingface.co/docs/hub/en/security-tokens).

The following models are _gated_. Verify that ``You have been granted access to this model`` appears on the model cards for any models you are interested in running locally:
  * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
  * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

Then, complete the following steps: 
1. If the project is already running, shut down the project environment under **Environment** > **Stop Environment**. This will ensure restarting the environment will incorporate all the below configurations. 
2. In AI Workbench, add the following entries under **Environment** > **Secrets**. 
   * <ins>Your Hugging Face Token</ins>: This is used to clone the model weights locally from Hugging Face.
       * _Name_: ``HUGGING_FACE_HUB_TOKEN``
       * _Value_: (Your HF API Key)
       * _Description_: HF Token for cloning model weights locally
3. **Rebuild** the project if needed to incorporate changes.

**Note:** All subsequent tutorials will assume both ``NVCF_RUN_KEY`` and ``HUGGING_FACE_HUB_TOKEN`` are already configured with your credentials. 

### Inference

1. Select the green **Open Chat** button on the top right the AI Workbench project window. 
2. Once the UI opens, click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
3. Select the **Local System** inference mode under ``Inference Settings`` > ``Inference Mode``. 
4. Select a model from the dropdown on the right hand settings panel. You can filter by gated vs ungated models for convenience. 
    * Ensure you have proper access permissions for the model; instructions are [here](https://github.com/NVIDIA/workbench-example-hybrid-rag?tab=readme-ov-file#additional-configurations).
    * You can also input a custom model from Hugging Face, following the same format. Careful, as not all models and quantization levels may be supported in the current TGI version!
5. Select a quantization level. The recommended precision for your system will be pre-selected for you, but full, 8-bit, and 4-bit bitsandbytes precision levels are currently supported. 

##### Table 2 System Resources vs Model Size and Quantization

 | vRAM    | System RAM | Disk Storage | Model Size & Quantization |
 |---------|------------|--------------|---------------------------|
 | >=12 GB | 32 GB      | 40 GB        | 7B & int4                 |
 | >=24 GB | 64 GB      | 40 GB        | 7B & int8                 |
 | >=40 GB | 64 GB      | 40 GB        | 7B & none                 |

6. Select **Load Model** to pre-fetch the model. This will take up to several minutes to perform an initial download of the model to the project cache. Subsequent loads will detect this cached model. 
7. Select **Start Server** to start the inference server with your current local GPU. This may take a moment to warm up.
8. Now, start chatting! Queries will be made to the model running on your local system whenever this inference mode is selected.

### Using RAG

9. In the right hand panel of the Chat UI select **Upload Documents Here**. Click to upload or drag and drop the desired text files to upload.
   * You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
10. Once the files upload, the **Toggle to Use Vector Database** next to the text input box will turn on by default.
11. Now query your documents! To use a different model, stop the server, make your selections, and restart the inference server. 

## Tutorial 2: Using a Remote Microservice
This tutorial assumes you already cloned this Hybrid RAG project to your AI Workbench. If not, please follow the beginning of the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint). 

<img src="./code/chatui/static/remote-ms.gif" width="85%" height="auto">

### Additional Configurations

* Set up your NVIDIA Inference Microservice (NIM) to run self-hosted on another system of your choice. The playbook to get started is located [here](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html). Remember the _model name_ (if not the ``meta/llama3-8b-instruct`` default) and the _ip address_ of this running microservice. Ports for NIMs are generally set to 8000 by default.
* **Alternatively**, you may set up any other 3rd party supporting the OpenAI API Specification. One example is [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building), as they support the [OpenAI API Spec](https://github.com/ollama/ollama/blob/main/docs/openai.md). Remember the _model name_, _port_, and the _ip address_ when you set this up. 

### Inference

1. Select the green **Open Chat** button on the top right the AI Workbench project window. 
2. Once the UI opens, click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
3. Select the **Self-hosted Microservice** inference mode under ``Inference Settings`` > ``Inference Mode``. 
4. Select the **Remote** tab in the right hand settings panel. Input the **IP address** of the accessible system running the microservice, **Port** if different from the 8000 default for NIMs, as well as the **model name** to run if different from the ``meta/llama3-8b-instruct`` default. 
5. Now start chatting! Queries will be made to the microservice running on a remote system whenever this inference mode is selected.

### Using RAG

6. In the right hand panel of the Chat UI select **Upload Documents Here**. Click to upload or drag and drop the desired text files to upload. 
   * You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
7. Once uploaded successfully, the **Toggle to Use Vector Database** should turn on by default next to your text input box.
8. Now you may query your documents!

## Tutorial 3: Using a Local Microservice
This tutorial assumes you already cloned this Hybrid RAG project to your AI Workbench. If not, please follow the beginning of the [Quickstart Tutorial](#tutorial-using-a-cloud-endpoint). 

<img src="./code/chatui/static/local-ms.gif" width="85%" height="auto">

Here are some important **PREREQUISITES**:
* Your AI Workbench <ins>must</ins> be running with a **DOCKER** container runtime. Podman is unsupported.
* You must have access to NeMo Inference Microservice (NIMs) [General Availability Program](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama3-8b-instruct/tags). 
* Shut down any other processes running locally on the GPU as these may result in memory issues when running the microservice locally. 

### Additional Configurations
Some additional configurations in AI Workbench are required to run this tutorial. Unlike the previous tutorials, these configs are not added to the project by default, so please follow the following instructions closely to ensure a proper setup. 

1. If running, shut down the project environment under **Environment** > **Stop Environment**. This will ensure restarting the environment will incorporate all the below configurations. 
2. In AI Workbench, add the following entries under **Environment** > **Secrets**:
   * <ins>Your NGC API Key</ins>: This is used to authenticate when pulling the NIM container from NGC. Remember, you must be in the General Availability Program to access this container.
       * _Name_: ``NGC_CLI_API_KEY``
       * _Value_: (Your NGC API Key)
       * _Description_: NGC API Key for NIM authentication
3. Add and/or modify the following under **Environment** > **Variables**:
   * ``DOCKER_HOST``: location of your docker socket, eg. ``unix:///var/host-run/docker.sock``
   * ``LOCAL_NIM_HOME``: location of where your NIM files will be stored, for example ``/mnt/c/Users/<my-user>`` for Windows or ``/home/<my-user>`` for Linux
4. Add the following under **Environment** > **Mounts**:
   * <ins>A Docker Socket Mount</ins>: This is a mount for the docker socket for the container to properly interact with the host Docker Engine.
      * _Type_: ``Host Mount``
      * _Target_: ``/var/host-run``
      * _Source_: ``/var/run``
      * _Description_: Docker socket Host Mount
   * <ins>A Filesystem Mount</ins>: This is a mount to properly run and manage your LOCAL_NIM_HOME on the host from inside the project container for generating the model repo. 
      * _Type_: ``Host Mount``
      * _Target_: ``/mnt/host-home``
      * _Source_: (Your LOCAL_NIM_HOME location) , for example ``/mnt/c/Users/<my-user>`` for Windows or ``/home/<my-user>`` for Linux
      * _Description_: Host mount for LOCAL_NIM_HOME
5. **Rebuild** the project if needed.

### Inference
1. Select the green **Open Chat** button on the top right the AI Workbench project window.
2. Once the UI opens, click **Set up RAG Backend**. This triggers a one-time backend build which can take a few moments to initialize.
3. Select the **Self-hosted Microservice** inference mode under ``Inference Settings`` > ``Inference Mode``. 
4. Select the **Local** sub-tab in the right hand settings panel.
5. Bring your **NIM Container Image** (placeholder can be used as the default flow), and select **Prefetch NIM**. This one-time process can take a few moments to pull down the NIM container.
6. Select **Start Microservice**. This may take a few moments to complete. 
7. Now, you can start chatting! Queries will be made to your microservice running on the local system whenever this inference mode is selected.

### Using RAG

8. In the right hand panel of the Chat UI select **Upload Documents Here**. Click to upload or drag and drop the desired text files to upload. 
   * You may see a warning that the vector database is not ready yet. If so wait a moment and try again. 
9. Once uploaded successfully, the **Toggle to Use Vector Database** should turn on by default next to your text input box.
10. Now you may query your documents!

## Tutorial 4: Customizing the Gradio App
By default, you may customize Gradio app using the jupyterlab container application. Alternatively, you may configure VSCode support [here](https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications/built-in/vs-code.html).

1. In AI Workbench, select the green **dropdown** from the top right and select **Open JupyterLab**.
2. Go into the `code/chatui/` folder and start editing the files.
3. Save the files.
4. To see your changes, stop the Chat UI and restart it.
5. To version your changes, commit them in the Workbench project window and push to your GitHub repo.

In addition to modifying the Gradio frontend, you can also use the Jupyterlab or another IDE to customize other aspects of the project, eg. custom chains, backend server, scripts, configs, etc.

# License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/LICENSE.txt)

This project may download and install additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 
