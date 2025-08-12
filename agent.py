import asyncio
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from github import Github
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult, ToolCall
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

load_dotenv()

# Initialize OpenAI LLM
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5",
    temperature=0.1
)

github_client = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
repo_url = "https://github.com/crazycabo/ai-engineer-projects_recipes-api.git"
pr_number = os.getenv("PR_NUMBER")

def parse_repo_info(url: str) -> tuple:
    """Extract owner and repo name from GitHub URL"""
    pattern = r"github\.com[/:]([^/]+)/([^/.]+)"
    match = re.search(pattern, url)

    if match:
        return match.group(1), match.group(2)
    raise ValueError("Invalid GitHub repository URL")

# Parse repo info from the URL
REPO_OWNER, REPO_NAME = parse_repo_info(repo_url)
repo = github_client.get_repo(f"{REPO_OWNER}/{REPO_NAME}")

def get_pr_details(pr_number: int) -> Dict:
    """
    Retrieve comprehensive details about a GitHub pull request.

    This tool fetches essential information about a specific pull request including
    metadata, author information, and associated commits. Use this when you need
    to understand the context and scope of a pull request.

    Args:
        pr_number (int): The unique identifier number of the pull request

    Returns:
        Dict: A dictionary containing:
            - author: GitHub username of the PR creator
            - title: The pull request title
            - body: Full description/body text of the PR
            - diff_url: Direct URL to view the unified diff
            - state: Current state (open, closed, merged)
            - commit_shas: List of all commit SHA hashes in this PR
    """
    # Get PR details
    pr = repo.get_pull(pr_number)

    # Get commit SHAs
    commits = pr.get_commits()
    commit_shas = [commit.sha for commit in commits]

    return {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_shas": commit_shas
    }

def get_commit_details(commit_sha: str) -> List[Dict]:
    """
    Analyze file-level changes within a specific commit.

    This tool provides detailed information about what files were modified,
    added, or deleted in a commit, along with the actual code changes.
    Essential for understanding the scope and impact of code changes.

    Args:
        commit_sha (str): The unique SHA hash identifier of the commit

    Returns:
        List[Dict]: A list of dictionaries, one for each modified file, containing:
            - filename: Path and name of the modified file
            - status: Type of change (added, modified, deleted, renamed)
            - additions: Number of lines added to the file
            - deletions: Number of lines removed from the file
            - changes: Total number of line changes
            - patch: The actual diff/patch showing line-by-line changes
    """
    # Get commit details
    commit = repo.get_commit(commit_sha)

    result = []
    for file in commit.files:
        file_info = {
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "changes": file.changes,
            "patch": file.patch if hasattr(file, 'patch') and file.patch else ""
        }
        result.append(file_info)

    return result

def get_file_contents(file_path: str, ref: str = "main") -> str:
    """
    Retrieve the complete contents of a file from the repository.

    This tool fetches the full text content of any file in the repository
    at a specific branch, tag, or commit. Useful for understanding the
    current state of code, documentation, or configuration files.

    Args:
        file_path (str): The relative path to the file within the repository
        ref (str, optional): Branch name, tag, or commit SHA to fetch from.
                            Defaults to "main" branch.

    Returns:
        str: The complete text content of the file
    """
    # Get file contents
    file_content = repo.get_contents(file_path, ref=ref)

    # Decode content (PyGitHub handles base64 decoding automatically)
    return file_content.decoded_content.decode('utf-8')

def create_pr_review(pr_number: int, comment_body: str) -> Dict:
    """
    Create and post a review comment on a GitHub pull request.

    This tool creates an actual review comment on the specified pull request
    using the GitHub API. The review will be posted immediately.

    Args:
        pr_number (int): The unique identifier number of the pull request
        comment_body (str): The markdown-formatted review comment text

    Returns:
        Dict: Information about the created review including:
            - message: Status message about the operation
            - review_id: ID of the created review (if successful)
            - review_url: URL to view the review (if successful)
    """
    try:
        pr = repo.get_pull(pr_number)
        review = pr.create_review(body=comment_body)

        return {
            "message": f"Review successfully posted on PR #{pr_number}",
            "review_id": review.id,
            "review_url": review.html_url
        }
    except Exception as e:
        return {
            "message": f"Failed to create review on PR #{pr_number}: {str(e)}"
        }

async def create_draft_pr_comment(comment_body: str, current_context: Context = None) -> str:
    """
    Create a draft pull request review comment.

    This tool creates a draft review comment for a specific pull request.
    The comment can be edited and submitted later through the GitHub interface.

    Args:
        comment_body (str): The markdown-formatted review comment text
        current_context (Context, optional): LlamaIndex Context object to store the draft comment
    """
    if current_context is not None:
        current_state = await current_context.store.get("state")
        current_state["draft_comment"] = comment_body
        await current_context.store.set("state", current_state)

    return "Created draft comment successfully. You can now edit and submit it in GitHub."

async def get_current_context() -> Dict:
    """
    Retrieve the current context gathered by the context gathering agent.

    This tool provides access to all the information that has been collected
    about the pull request, including PR details, changed files, and any
    additional repository files that were requested.

    Returns:
        Dict: A dictionary containing all gathered context information
    """
    return await ctx.store.get("state")

async def add_context_to_state(context_data: Dict, current_context: Context = None) -> str:
    """
    Add gathered context information to the LlamaIndex Context object.

    This tool allows the context gathering agent to store collected information
    about pull requests, commits, and files in the workflow's context that can be
    accessed by other agents.

    Args:
        context_data (Dict): Dictionary containing context information to store
        current_context (Context, optional): LlamaIndex Context object to store data in
    """
    if current_context is not None:
        for key, value in context_data.items():
            current_state = await current_context.store.get("state")
            current_state[key] = value
            await current_context.store.set("state", current_state)

    return "Context data added successfully to the workflow state."

# GitHub tool functions
pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)
file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
create_comment_tool = FunctionTool.from_defaults(fn=create_draft_pr_comment)
create_review_tool = FunctionTool.from_defaults(fn=create_pr_review)

# Context tool functions
get_context_tool = FunctionTool.from_defaults(fn=get_current_context)
add_context_tool = FunctionTool.from_defaults(fn=add_context_to_state)

# Define system prompt for the context gathering agent
CONTEXT_SYSTEM_PROMPT = """
You are the context gathering agent. When gathering context, you MUST gather: 
  - The details: author, title, body, diff_url, state, and head_sha
  - Changed files
  - Any requested for files
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""

# Create the ContextAgent as a FunctionAgent
context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers context from GitHub PRs and provides it to the CommentorAgent",
    tools=[pr_details_tool, commit_details_tool, file_contents_tool, get_context_tool, add_context_tool],
    llm=llm,
    system_prompt=CONTEXT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"]
)

# Update system prompt for the commentor agent to support handoff
COMMENTOR_SYSTEM_PROMPT = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the ContextAgent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. 
"""

# Create the CommentorAgent as a FunctionAgent
commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Writes review comments for GitHub PRs",
    tools=[create_comment_tool],
    llm=llm,
    system_prompt=COMMENTOR_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

# Define system prompt for the Review and Posting agent
REVIEW_AND_POSTING_SYSTEM_PROMPT = """
You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: 
   - Be a ~200-300 word review in markdown format. 
   - Specify what is good about the PR: 
   - Did the author follow ALL contribution rules? What is missing? 
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? 
   - Are there notes on whether new endpoints were documented? 
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? 
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. 
 When you are satisfied, post the review to GitHub.
"""

# Create the ReviewAndPostingAgent as a FunctionAgent
review_and_posting_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description="Reviews generated comments and posts them to GitHub PRs",
    tools=[create_review_tool, add_context_tool],
    llm=llm,
    system_prompt=REVIEW_AND_POSTING_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"]
)

# Create the AgentWorkflow
workflow_agent = AgentWorkflow(
    agents=[review_and_posting_agent, commentor_agent, context_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": "",
    }
)

ctx = Context(workflow_agent)

async def main():
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query.strip())

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

if __name__ == "__main__":
    asyncio.run(main())
    github_client.close()
