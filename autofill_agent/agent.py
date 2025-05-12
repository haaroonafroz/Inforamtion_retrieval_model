import os
import asyncio
from typing import List, Dict, Optional, Any

# --- ADK Imports ---
from google.adk.agents import Agent
# Potentially Tool class if needed for advanced configuration, but functions usually suffice
# from google.adk.tools import Tool

# --- Langchain/PDF Imports ---
from langchain.schema import Document as LangchainDocument

# --- Playwright Imports ---
# Make sure playwright is installed: pip install playwright && playwright install
from playwright.async_api import async_playwright, Playwright, Browser, Page

# --- Custom Module Imports ---
# Use relative imports assuming agent.py is inside autofill_agent package
from .load_and_process_pdf import load_and_split_pdf
from .retrieve_info_from_pdf import RAGManager
from .analyze_web_form import analyze_form_structure
from .interact_with_web_page import BrowserInteractor

# --- Global State / Resource Management ---
# Warning: Global state can be tricky in long-running/multi-session apps.
# Consider more robust state management (e.g., class-based agent, session state) for production.
rag_manager: Optional[RAGManager] = None
pdf_chunks: Optional[List[LangchainDocument]] = None
playwright_context: Optional[Playwright] = None
browser_context: Optional[Browser] = None
page_context: Optional[Page] = None
browser_interactor: Optional[BrowserInteractor] = None
current_form_fields: Optional[List[Dict[str, Any]]] = None
current_page_html: Optional[str] = None
processed_pdf_path: Optional[str] = None


# --- Tool Definitions (Wrapping Custom Functions/Methods) ---

async def setup_browser_tool() -> str:
    """
    Initializes the web browser using Playwright if not already initialized.
    This MUST be called before any web interaction tools.
    Returns:
        A status message indicating success or failure.
    """
    global playwright_context, browser_context, page_context, browser_interactor
    if page_context and not page_context.is_closed():
        return "Browser already initialized and page is open."
    try:
        print("Initializing Playwright browser...")
        playwright_context = await async_playwright().start()
        # Use chromium, or change to 'firefox' or 'webkit' if needed
        browser_context = await playwright_context.chromium.launch(headless=True) # Use False to see browser
        page_context = await browser_context.new_page()
        browser_interactor = BrowserInteractor(page_context)
        print("Browser initialized successfully.")
        return "Browser initialized successfully."
    except Exception as e:
        print(f"Error initializing browser: {e}")
        return f"Error initializing browser: {e}"

async def close_browser_tool() -> str:
    """
    Closes the Playwright browser and cleans up resources.
    Should be called when web interactions are finished.
    Returns:
        A status message indicating success or failure.
    """
    global playwright_context, browser_context, page_context, browser_interactor
    closed = False
    try:
        if page_context and not page_context.is_closed():
            await page_context.close()
            print("Closed Playwright page.")
            closed = True
        if browser_context and browser_context.is_connected():
            await browser_context.close()
            print("Closed Playwright browser.")
            closed = True
        if playwright_context:
            await playwright_context.stop()
            print("Stopped Playwright.")
            closed = True

        page_context = None
        browser_context = None
        playwright_context = None
        browser_interactor = None

        if closed:
            return "Browser resources closed successfully."
        else:
            return "No active browser resources found to close."
    except Exception as e:
        print(f"Error closing browser resources: {e}")
        return f"Error closing browser resources: {e}"


def load_process_and_index_pdf_tool(pdf_path: str) -> str:
    """
    Loads a PDF file, splits it into chunks, and initializes the RAG system
    to allow querying the PDF content. This must be called before querying PDF info.
    Args:
        pdf_path: The file system path to the PDF document.
    Returns:
        A status message indicating success or failure and the number of chunks processed.
    """
    global rag_manager, pdf_chunks, processed_pdf_path
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at '{pdf_path}'"
    try:
        print(f"Processing PDF: {pdf_path}")
        # 1. Load and Split
        # Use defaults or make chunk_size/overlap configurable via agent args/tools if needed
        pdf_chunks = load_and_split_pdf(pdf_file_path=pdf_path)

        # 2. Initialize RAG Manager (adjust config as needed)
        # Persist directory could be './output/vector_store' based on your structure
        persist_dir = os.path.join('output', 'vector_store')
        # Ensure the directory exists if using persistence
        os.makedirs(persist_dir, exist_ok=True)

        rag_manager = RAGManager(persist_directory=persist_dir) # Or None for in-memory

        # 3. Initialize Vector Store (force_recreate=True ensures fresh index for this PDF)
        rag_manager.initialize_vector_store(chunks=pdf_chunks, force_recreate=True)
        processed_pdf_path = pdf_path # Store path for context
        chunk_count = len(pdf_chunks) if pdf_chunks else 0
        print(f"PDF processed and indexed. Found {chunk_count} chunks.")
        return f"Successfully processed and indexed PDF '{os.path.basename(pdf_path)}'. {chunk_count} chunks indexed."

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        rag_manager = None # Reset on error
        pdf_chunks = None
        processed_pdf_path = None
        return f"Error processing PDF {pdf_path}: {e}"

def query_pdf_content_tool(query: str) -> List[str]:
    """
    Queries the content of the previously processed PDF using the RAG system.
    Use this to find specific details needed for form fields (e.g., 'first name', 'email address').
    Args:
        query: The specific information or question to ask the PDF content.
    Returns:
        A list of relevant text snippets found in the PDF, or an empty list if none found or RAG not ready.
    """
    global rag_manager
    if rag_manager is None or rag_manager.vector_store is None:
        return ["Error: RAG system not initialized. Process a PDF first."]

    print(f"Querying PDF content for: '{query}'")
    try:
        # Retrieve relevant documents (adjust k as needed)
        results: Optional[List[LangchainDocument]] = rag_manager.query_vector_store(query=query, k=3)

        if results:
            print(f"Found {len(results)} relevant chunks.")
            return [doc.page_content for doc in results]
        else:
            print("No relevant information found in PDF for this query.")
            return []
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        return [f"Error during PDF query: {e}"]

async def navigate_and_get_html_tool(url: str) -> str:
    """
    Navigates the initialized browser to the specified URL and returns the page's full HTML content.
    Args:
        url: The web address (URL) to navigate to.
    Returns:
        The HTML content of the page as a string, or an error message.
    """
    global browser_interactor, page_context, current_page_html
    if browser_interactor is None or page_context is None or page_context.is_closed():
        return "Error: Browser not initialized or page closed. Call setup_browser_tool first."
    try:
        print(f"Navigating to URL: {url}")
        await page_context.goto(url, wait_until='domcontentloaded') # Or 'networkidle'
        print(f"Navigation successful. Getting page content...")
        html_content = await browser_interactor.get_page_content()
        if html_content:
            current_page_html = html_content # Store for analysis
            print(f"Retrieved HTML content ({len(html_content)} bytes).")
            # Return only a snippet or confirmation to avoid overwhelming the LLM context
            return f"Successfully navigated to {url} and retrieved HTML content."
            # Or return html_content if analysis tool needs it directly (might exceed context limits)
        else:
            current_page_html = None
            return f"Successfully navigated to {url}, but failed to retrieve HTML content."
    except Exception as e:
        print(f"Error navigating to or getting content from {url}: {e}")
        current_page_html = None
        return f"Error navigating or getting content: {e}"

def analyze_web_form_tool() -> List[Dict[str, Any]]:
    """
    Analyzes the HTML content of the *current* page (retrieved by navigate_and_get_html_tool)
    to identify form fields like inputs, textareas, selects, and buttons.
    Returns:
        A list of dictionaries, each describing a form field found (keys: 'label', 'type', 'selector', 'id', 'name', 'options').
        Returns an empty list if no HTML is available or no fields are found.
    """
    global current_page_html, current_form_fields
    if current_page_html is None:
        return [{"error": "No HTML content available. Use navigate_and_get_html_tool first."}]

    print("Analyzing current page HTML for form structure...")
    try:
        form_fields = analyze_form_structure(current_page_html)
        current_form_fields = form_fields # Store for filling
        print(f"Form analysis complete. Found {len(form_fields)} potential fields/buttons.")
        # Return the structured data for the LLM to reason about
        return form_fields
    except Exception as e:
        print(f"Error analyzing form structure: {e}")
        current_form_fields = None
        return [{"error": f"Failed to analyze form structure: {e}"}]


async def fill_web_form_field_tool(selector: str, value: str) -> str:
    """
    Fills a specific input field on the current webpage, identified by its CSS selector, with the provided value.
    Args:
        selector: The CSS selector (e.g., '#firstname', '[name="email"]') identifying the input field.
        value: The text value to fill into the field.
    Returns:
        A status message indicating success or failure.
    """
    global browser_interactor
    if browser_interactor is None:
        return "Error: Browser not initialized. Call setup_browser_tool first."

    print(f"Attempting to fill field '{selector}' with value '{value[:50]}...'")
    success = await browser_interactor.fill_field(selector=selector, value=value)
    if success:
        return f"Successfully filled field '{selector}'."
    else:
        return f"Failed to fill field '{selector}'."

async def select_dropdown_option_tool(selector: str, option_text: str) -> str:
    """
    Selects an option within a dropdown (<select>) element on the current webpage.
    It tries to match the visible text of the option.
    Args:
        selector: The CSS selector for the <select> element.
        option_text: The visible text of the option to select.
    Returns:
        A status message indicating success or failure.
    """
    global browser_interactor
    if browser_interactor is None:
        return "Error: Browser not initialized. Call setup_browser_tool first."

    print(f"Attempting to select option '{option_text}' in dropdown '{selector}'")
    # Using 'label' which corresponds to the visible text
    success = await browser_interactor.select_dropdown_option(selector=selector, label=option_text)
    if success:
        return f"Successfully selected option '{option_text}' in '{selector}'."
    else:
        return f"Failed to select option '{option_text}' in '{selector}'."

async def check_web_checkbox_tool(selector: str, should_be_checked: bool) -> str:
    """
    Checks or unchecks a checkbox on the current webpage.
    Args:
        selector: The CSS selector for the checkbox input element.
        should_be_checked: True to check the box, False to uncheck it.
    Returns:
        A status message indicating success or failure.
    """
    global browser_interactor
    if browser_interactor is None:
        return "Error: Browser not initialized. Call setup_browser_tool first."

    action = "check" if should_be_checked else "uncheck"
    print(f"Attempting to {action} checkbox '{selector}'")
    success = await browser_interactor.set_checkbox(selector=selector, check=should_be_checked)
    if success:
        return f"Successfully {action}ed checkbox '{selector}'."
    else:
        return f"Failed to {action} checkbox '{selector}'."

async def click_web_element_tool(selector: str) -> str:
    """
    Clicks an element on the current webpage, identified by its CSS selector.
    Useful for clicking buttons (like 'Submit'), links, etc.
    Args:
        selector: The CSS selector for the element to click.
    Returns:
        A status message indicating success or failure.
    """
    global browser_interactor
    if browser_interactor is None:
        return "Error: Browser not initialized. Call setup_browser_tool first."

    print(f"Attempting to click element '{selector}'")
    success = await browser_interactor.click_element(selector=selector)
    if success:
        # After clicking, the page might change, invalidating old HTML/fields
        global current_page_html, current_form_fields
        current_page_html = None
        current_form_fields = None
        print(f"Element '{selector}' clicked. Page state may have changed.")
        return f"Successfully clicked element '{selector}'."
    else:
        return f"Failed to click element '{selector}'."


# --- Agent Definition ---

# Combine all defined tools into a list
autofill_tools = [
    setup_browser_tool,
    load_process_and_index_pdf_tool,
    query_pdf_content_tool,
    navigate_and_get_html_tool,
    analyze_web_form_tool,
    fill_web_form_field_tool,
    select_dropdown_option_tool,
    check_web_checkbox_tool,
    click_web_element_tool,
    close_browser_tool, # Important for cleanup
]

# Define the main agent using the Gemini model and the tools
autofill_agent = Agent(
    name="autofill_manager_agent",
    model="gemini-2.0-flash", # Or "gemini-1.5-flash-001" etc. depending on availability/preference
    # model="gemini-1.5-pro-latest", # Consider Pro for more complex reasoning if needed

    instruction="""You are an AI assistant designed to automatically fill web forms using information provided in a PDF document.

    **Workflow:**
    1.  **Receive Goal:** The user will provide the path to a PDF file and the URL of a web form.
    2.  **Initialize:** Call `setup_browser_tool` to start the web browser.
    3.  **Process PDF:** Call `load_process_and_index_pdf_tool` with the provided PDF path. This makes the PDF content searchable.
    4.  **Navigate:** Call `Maps_and_get_html_tool` with the target URL to load the webpage.
    5.  **Analyze Form:** Call `analyze_web_form_tool` to get a list of all form fields on the current page. Pay attention to the 'label', 'type', and 'selector' for each field.
    6.  **Fill Fields (Iterative Process):**
        * For each field identified in the analysis:
            * Determine the information needed based on its 'label' or 'name' (e.g., "First Name", "Email", "Country").
            * Call `query_pdf_content_tool` with a specific query to find that information in the PDF (e.g., "What is the user's first name?").
            * Based on the field 'type' and the retrieved information:
                * If it's a text input (`text`, `email`, `tel`, etc.) or `textarea`, use `fill_web_form_field_tool` with the field's 'selector' and the retrieved value.
                * If it's a `select` (dropdown), examine its 'options' (from analysis) and use `select_dropdown_option_tool` with the 'selector' and the *exact text* of the best matching option based on the PDF info.
                * If it's a `checkbox`, decide if it should be checked based on the PDF info (this might require careful querying) and use `check_web_checkbox_tool` with the 'selector' and True/False.
                * Handle `radio` buttons similarly if needed (may require a dedicated tool or careful use of `click_web_element_tool`).
        * **IMPORTANT:** Only fill fields for which you confidently found information in the PDF. Do not guess or fill with placeholder text. If multiple relevant snippets are returned from the PDF query, use the most likely one or ask for clarification if ambiguous.
    7.  **Submit (Optional):** After attempting to fill all fields, look for a submit button (check field analysis results for type 'submit' or 'button' with relevant text/label). If found, *ask the user* if you should click it using `click_web_element_tool`. Do not click submit automatically.
    8.  **Cleanup:** Once the process is complete or if instructed to stop, call `close_browser_tool` to release resources.

    **Error Handling:** If any step fails, report the error clearly and stop the process unless the error is recoverable (e.g., trying a different query). If the browser needs setup, call `setup_browser_tool`. If the PDF isn't processed, call `load_process_and_index_pdf_tool`.
    """,
    description="An agent that automatically fills web forms using data extracted from a PDF file via RAG and web interaction via Playwright.",
    tools=autofill_tools,
    # enable_tool_check=True # Can add extra validation layer if needed
)

# --- Optional: Main block for testing directly (if not using adk run/web) ---
# async def run_test():
#     # Example of how you might invoke it programmatically (simplified)
#     # Note: ADK's Runner/SessionService provides the proper execution context
#     print("Starting agent test...")
#     pdf_test_path = "path/to/your/test.pdf" # <--- CHANGE THIS
#     url_test = "https://example-form.com"   # <--- CHANGE THIS

#     # Simulate user input or initial message
#     initial_prompt = f"Please autofill the form at {url_test} using the information in the PDF at {pdf_test_path}."

#     # This is a conceptual run, ADK handles the actual event loop
#     # You'd typically use adk.runners.Runner and adk.sessions.SessionService

#     print("Agent defined. Use 'adk web' or 'adk run autofill_agent' to interact.")
#     # Cleanup might need manual trigger if not run via ADK runner lifecycle
#     # await close_browser_tool()


# if __name__ == "__main__":
#     # To run the conceptual test (requires manual setup of pdf/url)
#     # asyncio.run(run_test())
#     print(f"Agent 'autofill_agent' is defined in {__file__}")
#     print("Run using 'adk web' or 'adk run autofill_agent'.")