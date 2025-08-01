# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel

# Adjust the sys.path so we can use the GitHubPlugin and GitHubSettings classes
# This is so we can run the code from the samples/learn_resources/agent_docs directory
# If you are running code from your own project, you may not need need to do this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plugins.GithubPlugin.github import GitHubPlugin, GitHubSettings  # noqa: E402

"""
The following sample demonstrates how to create a simple,
ChatCompletionAgent to use a GitHub plugin to interact
with the GitHub API.

This is the full code sample for the Semantic Kernel Learn Site: How-To: Chat Completion Agent

https://learn.microsoft.com/semantic-kernel/frameworks/agent/examples/example-chat-agent?pivots=programming-language-python
"""

#load variables
load_dotenv()

# Variables - Azure Services
azopenai_ep=os.environ["AZURE_OPENAI_ACCOUNT"]
azopenai_key=os.environ["AZURE_OPENAI_KEY"]
azopenai_model=os.environ["AZURE_OPENAI_MODEL"]
github_token=os.environ["GITHUB_TOKEN"]

async def main():
    kernel = Kernel()

    # Add the AzureChatCompletion AI Service to the Kernel
    service_id = "agent"
    
    #kernel.add_service(AzureChatCompletion(service_id=service_id))
    
    # Add Azure OpenAI chat completion
    # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.azurechatcompletion?view=semantic-kernel-python
    chat_completion = AzureChatCompletion(
        deployment_name=azopenai_model,
        api_key=azopenai_key,
        endpoint=azopenai_ep,
        service_id=service_id
    )
    
    # Add the chat completion service to the kernel
    kernel.add_service(chat_completion)
    
    # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.services.kernel_services_extension.kernelservicesextension?view=semantic-kernel-python#semantic-kernel-services-kernel-services-extension-kernelservicesextension-get-prompt-execution-settings-from-service-id
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    # Configure the function choice behavior to auto invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Set your GitHub Personal Access Token (PAT) value here
    gh_settings = GitHubSettings(token=github_token)  
    kernel.add_plugin(plugin=GitHubPlugin(gh_settings), plugin_name="GithubPlugin")

    current_time = datetime.now().isoformat()

    # Create the agent definition
    # Instantiate a ChatCompletionAgent with its Instructions, associated Kernel, and the default Arguments and Execution Settings. In this case, we desire to have the any plugin functions automatically executed.
    # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.agents.chat_completion.chat_completion_agent.chatcompletionagent?view=semantic-kernel-python
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="SampleAssistantAgent",
        instructions=f"""
            You are an agent designed to query and retrieve information from a single GitHub repository in a read-only 
            manner.
            You are also able to access the profile of the active user.

            Use the current date and time to provide up-to-date details or time-sensitive responses.
            
            The repository you are querying is a public repository with the following name: pradorodriguez/tc25-iims

            The current date and time is: {current_time}. 
            """,
        # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.functions.kernel_arguments.kernelarguments?view=semantic-kernel-python        
        arguments=KernelArguments(settings=settings),
    )

    # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.agents.chat_completion.chat_completion_agent.chathistoryagentthread?view=semantic-kernel-python
    thread: ChatHistoryAgentThread = None
    is_complete: bool = False
    while not is_complete:
        user_input = input("User:> ")
        if not user_input:
            continue

        if user_input.lower() == "exit":
            is_complete = True
            break

        arguments = KernelArguments(now=datetime.now().strftime("%Y-%m-%d %H:%M"))

        # https://learn.microsoft.com/en-us/python/api/semantic-kernel/semantic_kernel.agents.agent.agent?view=semantic-kernel-python
        async for response in agent.invoke(messages=user_input, thread=thread, arguments=arguments):
            print(f"{response.content}")
            thread = response.thread


if __name__ == "__main__":
    asyncio.run(main())