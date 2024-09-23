# Finexial [![Open In AI Workbench](https://img.shields.io/badge/Open_In-AI_Workbench-76B900)](https://github.com/AmandineFlachs/finexial.git)
### Finexial is an AI-powered tool designed to understand financial reports, discuss reports' outcome and explain any financial jargon.

## Project summary 

<!-- Banner Image 
<img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/07/rag-representation.jpg" width="100%">-->

<!-- Links
<p align="center"> 
  <a href="https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/" style="color: #76B900;">:arrow_down: Download AI Workbench</a> •
  <a href="https://docs.nvidia.com/ai-workbench/" style="color: #76B900;">:book: Read the Docs</a> •
  <a href="https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html" style="color: #76B900;">:open_file_folder: Explore Example Projects</a> •
  <a href="https://forums.developer.nvidia.com/t/support-workbench-example-project-agentic-rag/303414" style="color: #76B900;">:rotating_light: Facing Issues? Let Us Know!</a>
</p> -->

Companies expect 70% of their employee to heavily use data to make decisions by 2025, a jump of over 40% since 2018 (Tableau, 2021). Still, 74% of employees report feeling overwhelmed when working with data (Accenture, 2020). While a majority of business decisions are influenced by financial data, only 30% of employees from non-financial departments feel confident in their ability to understand and use financial information effectively (Deloitte, 2021). In the USA only, the lack of data and financial literacy is estimated to cost companies over $100B per year (Accenture, 2020).

Finexial is an AI RAG project created for the [HackAI - Dell and NVIDIA Challenge](https://hackaichallenge.devpost.com). 
It aims to help non-financial employees make the most of their financial data and feel more confident working with large financial reports.

## Repository

The following files are included in the repository

IMAGE TO ADD

## Set up and run Finexial

Finexial has been designed to be easy to run and use thanks to [Nvidia AI Workbench](https://www.nvidia.com/en-gb/deep-learning-ai/solutions/data-science/workbench/). It currently relies on the non-gated [nvidia/Llama3-ChatQA-1.5-8B](https://build.nvidia.com/nvidia/chatqa-1-5-8b) model. 

### Follow the steps to set up Finxial:
* Install and configure [Nvidia AI Workbench](https://www.nvidia.com/en-gb/deep-learning-ai/solutions/data-science/workbench/) locally and open up AI Workbench [(see the documentation for more info)](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html). NVIDIA AI Workbench is a free client application that handles Git repositories and containers and manages dev environments. 
* Fork this repository into your own GitHub account.
* Inside Nvidia AI Workbench:
    * Click ``CLONE PROJECT`` and enter the GitHub repository URL of your newly-forked repo.
    * Nvidia AI Workbench will automatically clone the repo and build out the project environment, which can take a while to complete.
    * In the Environment tab, add your own token (key) in the Secret section for both the ``HUGGING_FACE_HUB_TOKEN`` and ``NVCF_RUN_KEY``. You will need to create a free account respectively on [Hugging Face](https://huggingface.co) and [NVIDIA NGC](https://ngc.nvidia.com/signin) even if you plan to use Finexial locally.
    * Still in the Environment tab, click on ``START ENVIRONMENT``.
    * Then click on ``START CHAT`` to launch the application. A new tab in your web browser will automatically open. 

You are now ready to go!

## How to use Finexial

Now that the system is set up, you can run Finexial:
1. Choose your inference mode (local or cloud).
2. Click on the button ``START FINEXIAL``.
3. Upload your document(s) in pdf.
4. Use the chat box to ask questions about the reports and documents uploaded using natural language. At any time you can start again from scratch by clearing your chat history (``CLEAR HISTORY`` button) or remote the document(s) uploaded (``CLEAR DATABASE`` button). 

### Hardware requirements for local inference

While there are no hardware requirements for cloud inference, if you are running Finexial locally you will need to ensure you have sufficient GPU resources. For this project, I used an Nvidia RTX 3090. 

## How it works

IMAGE TECH ARCHITECTURE
    
# License
This project is under the Apache 2.0 License.

This project may download and install additional third-party open source software projects. Review the license terms of these open source projects before use. Third party components used as part of this project are subject to their separate legal notices or terms that accompany the components. You are responsible for confirming compliance with third-party component license terms and requirements. 
