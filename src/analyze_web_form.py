"""
Tool for analyzing the current web page form structure.
"""

from bs4 import BeautifulSoup
import json # Or return Pydantic models for better structure

# Potential form field types to identify
INPUT_TYPES = ["text", "password", "email", "number", "tel", "url", "date", "search"]

def find_label_for_element(element, soup):
    """Tries to find the associated <label> for a form element."""
    # 1. Check for label wrapping the element
    parent_label = element.find_parent('label')
    if parent_label:
        return parent_label.get_text(strip=True)

    # 2. Check for <label for=...> matching element's id
    element_id = element.get('id')
    if element_id:
        label = soup.find('label', {'for': element_id})
        if label:
            return label.get_text(strip=True)
    
    # 3. Simple proximity check (less reliable) - look for preceding text
    # This needs refinement for complex layouts
    prev_sibling = element.find_previous_sibling()
    if prev_sibling and prev_sibling.name != 'label': # Avoid double counting
         # Check if it's simple text or a container with text
        text = prev_sibling.get_text(strip=True)
        if text and len(text) < 100: # Avoid grabbing large text blocks
             # Further heuristics could be added (e.g., check for colon :) 
             return text

    # 4. Check aria-label or aria-labelledby (Accessibility)
    aria_label = element.get('aria-label')
    if aria_label:
        return aria_label
    aria_labelledby = element.get('aria-labelledby')
    if aria_labelledby:
        labelled_by_element = soup.find(id=aria_labelledby)
        if labelled_by_element:
            return labelled_by_element.get_text(strip=True)

    # 5. Use placeholder as a fallback label
    placeholder = element.get('placeholder')
    if placeholder:
        return placeholder # Indicate it's a placeholder?

    # 6. Use name attribute as last resort
    name = element.get('name')
    if name:
        return name # Indicate it's a name attribute?

    return None # Could not find a label

def analyze_form_structure(page_content: str) -> list[dict]:
    """
    Parses HTML content to find forms and extract their field structure.

    Args:
        page_content: The HTML content of the web page.

    Returns:
        A list of dictionaries, where each dictionary represents a form field
        containing info like 'label', 'type', 'selector', 'id', 'name'.
        Returns an empty list if no suitable form elements are found.
    """
    soup = BeautifulSoup(page_content, 'html.parser')
    forms = soup.find_all('form')
    extracted_fields = []

    # If no <form> tag, search the whole body for inputs (less ideal)
    search_area = forms if forms else [soup.body]
    if not search_area or search_area[0] is None:
        search_area = [soup] # Fallback to whole document if no body

    field_counter = 0
    for area in search_area:
        if area is None: continue
        # Find standard inputs
        inputs = area.find_all('input', {'type': lambda x: x in INPUT_TYPES or x is None})
        # Find textareas
        textareas = area.find_all('textarea')
        # Find select dropdowns
        selects = area.find_all('select')
        # Find checkboxes and radio buttons
        checkboxes = area.find_all('input', {'type': 'checkbox'})
        radios = area.find_all('input', {'type': 'radio'})
        # Find buttons (especially submit)
        buttons = area.find_all('button')
        submit_inputs = area.find_all('input', {'type': 'submit'})

        all_elements = inputs + textareas + selects + checkboxes + radios + buttons + submit_inputs

        for element in all_elements:
            field_info = {}
            tag_name = element.name
            element_type = element.get('type', tag_name) # Default to tag name if type absent
            
            # Generate a unique-enough selector (robust selector generation is complex)
            # Using id if available, otherwise name, or a basic tag[type] selector
            element_id = element.get('id')
            element_name = element.get('name')
            selector = f"#{element_id}" if element_id else (
                f"[name='{element_name}']" if element_name else (
                    f"{tag_name}[type='{element_type}']" if element.has_attr('type') else tag_name
                )
            )
            # Potential improvement: More specific CSS selector generation
            
            field_info['selector'] = selector
            field_info['type'] = element_type
            field_info['id'] = element_id
            field_info['name'] = element_name
            field_info['label'] = find_label_for_element(element, soup) or f"field_{field_counter}" # Fallback label
            
            if tag_name == 'select':
                field_info['options'] = [opt.get_text(strip=True) for opt in element.find_all('option')]

            extracted_fields.append(field_info)
            field_counter += 1
            
    print(f"Analyzed form structure, found {len(extracted_fields)} potential fields/buttons.")
    return extracted_fields

# Example Usage (for testing)
if __name__ == '__main__':
    # Example HTML content (replace with actual content fetched via Playwright)
    test_html = """
    <html><body>
        <form>
            <label for="fname">First name:</label><br>
            <input type="text" id="fname" name="fname" placeholder="John"><br>
            <label>Last name:
                <input type="text" name="lname" aria-label="Surname">
            </label><br>
            <input type="checkbox" id="vehicle1" name="vehicle1" value="Bike">
            <label for="vehicle1"> I have a bike</label><br>
            <select name="cars" id="cars">
                <option value="volvo">Volvo</option>
                <option value="saab">Saab</option>
            </select><br>
            <textarea name="message" rows="3" cols="30" placeholder="Your message"></textarea><br>
            <button type="submit">Submit</button>
        </form>
    </body></html>
    """
    
    print("--- Testing Form Analysis ---")
    analysis_result = analyze_form_structure(test_html)
    
    print("\n--- Analysis Result ---")
    print(json.dumps(analysis_result, indent=2)) 