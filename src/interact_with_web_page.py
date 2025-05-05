"""
Tool for interacting with web page elements using Playwright, encapsulated in a class.
"""

# This code assumes Playwright is installed and browser binaries are available.
# You might need to run 'playwright install' first.

# Import async Page type hint
from playwright.async_api import Page as AsyncPage 
from typing import Optional # For type hinting

class BrowserInteractor:
    """Encapsulates Playwright page interactions for use as agent tools."""

    def __init__(self, page: AsyncPage):
        """
        Initializes the interactor with an active Playwright Page object.

        Args:
            page: An initialized and potentially navigated Playwright async Page object.
        """
        if page is None:
            raise ValueError("BrowserInteractor must be initialized with a valid Playwright Page object.")
        self.page = page
        print(f"BrowserInteractor initialized with page: {page.url}")

    # --- Interaction Methods (Tools) --- 

    async def fill_field(self, selector: str, value: str) -> bool:
        """Fills a text input field identified by a selector with the given value."""
        try:
            print(f"Attempting to fill field '{selector}' with value '{value[:30]}...'")
            await self.page.locator(selector).fill(value)
            print(f"Successfully filled '{selector}'")
            return True
        except Exception as e:
            print(f"Error filling field '{selector}': {e}")
            return False # Indicate failure

    async def click_element(self, selector: str) -> bool:
        """Clicks an element identified by a selector."""
        try:
            print(f"Attempting to click element '{selector}'")
            # Add potential waits or visibility checks if needed
            # await self.page.locator(selector).wait_for(state="visible")
            await self.page.locator(selector).click()
            print(f"Successfully clicked '{selector}'")
            return True
        except Exception as e:
            print(f"Error clicking element '{selector}': {e}")
            return False

    async def set_checkbox(self, selector: str, check: bool = True) -> bool:
        """Checks (if check=True) or unchecks (if check=False) a checkbox identified by a selector."""
        try:
            action = "check" if check else "uncheck"
            print(f"Attempting to {action} checkbox '{selector}'")
            await self.page.locator(selector).set_checked(checked=check)
            print(f"Successfully {action}ed '{selector}'")
            return True
        except Exception as e:
            print(f"Error setting checkbox '{selector}': {e}")
            return False

    async def select_dropdown_option(self, selector: str, value: Optional[str] = None, label: Optional[str] = None) -> bool:
        """Selects an option in a dropdown (<select> element) identified by a selector. Select by 'value' attribute or by visible 'label' text."""
        if not value and not label:
            print("Error: Must provide either 'value' or 'label' for select_dropdown_option.")
            return False
            
        try:
            target = {'value': value} if value is not None else {'label': label}
            target_desc = f"value '{value}'" if value is not None else f"label '{label}'"
            print(f"Attempting to select option with {target_desc} in dropdown '{selector}'")
            await self.page.locator(selector).select_option(**target)
            print(f"Successfully selected option in '{selector}'")
            return True
        except Exception as e:
            print(f"Error selecting option in dropdown '{selector}': {e}")
            return False

    async def get_page_content(self) -> Optional[str]:
        """Gets the full HTML content of the current page associated with this interactor."""
        try:
            print(f"Attempting to get page content for: {self.page.url}")
            content = await self.page.content()
            print("Successfully retrieved page content.")
            return content
        except Exception as e:
            print(f"Error getting page content: {e}")
            return None

# --- Example Usage (Conceptual - requires running Playwright setup externally) --- 
async def main_test():
    # This block needs actual Playwright setup to run.
    # It demonstrates how to use the BrowserInteractor class.
    print("--- Testing BrowserInteractor (Conceptual) ---")
    
    playwright = None
    browser = None
    page = None
    interactor = None

    try:
        # 1. Setup Playwright (example)
        from playwright.async_api import async_playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True) # Use headless=False to see the browser
        page = await browser.new_page()
        
        # 2. Navigate to a test page (e.g., a local HTML file with a form)
        # For a real test, create a simple form.html file
        # Or navigate to a known website, e.g., "https://example.com"
        # await page.goto("file:///path/to/your/form.html") 
        await page.goto("data:text/html," + 
                       "<html><body><form>" +
                       "<label for='fname'>First name:</label><input type='text' id='fname' name='fname'><br>" +
                       "<input type='checkbox' id='chk' name='chk'><label for='chk'>Check me</label><br>" +
                       "<button type='button' id='btn'>Click Me</button>" +
                       "</form></body></html>")
        print(f"Navigated to: {page.url}")

        # 3. Initialize the interactor
        interactor = BrowserInteractor(page)

        # 4. Use the interactor methods
        html_content = await interactor.get_page_content()
        if html_content:
            print(f"Got HTML content (first 100 chars): {html_content[:100]}...")
        
        print("\nAttempting interactions...")
        success_fill = await interactor.fill_field("#fname", "Test Name")
        print(f"Fill success: {success_fill}")
        
        success_check = await interactor.set_checkbox("#chk", True)
        print(f"Check success: {success_check}")
        
        success_click = await interactor.click_element("#btn")
        print(f"Click success: {success_click}")
        
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 5. Cleanup
        if page:
            await page.close()
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()
        print("\nConceptual test finished and Playwright cleanup attempted.")

if __name__ == '__main__':
    import asyncio
    # Running the async test requires an event loop
    # asyncio.run(main_test())
    print("Refactored to BrowserInteractor class. Run the main_test function within an async context to test interactively.")
    # Example of how to run it if needed:
    # try:
    #     asyncio.run(main_test())
    # except KeyboardInterrupt:
    #     print("Test interrupted.") 