import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from xml.etree.ElementTree import Element, SubElement, tostring, fromstring


class CustomMarkdownProcessor(Treeprocessor):
    def run(self, root):
        """Process Markdown-generated HTML"""
        print("DEBUG: Running CustomMarkdownProcessor...")
        
        self.wrap_tables(root)
        self.modify_code_blocks(root)

    def wrap_tables(self, root):
        """Wrap <table> elements inside <figure> for better styling."""
        tables = list(root.findall(".//table"))
        for table in tables:
            figure = Element("figure")
            parent = self.find_parent(root, table)
            if parent:
                index = list(parent).index(table)
                parent.remove(table)
                figure.append(table)
                parent.insert(index, figure)

    def modify_code_blocks(self, root):
        """Wrap <pre><code> blocks with <div> and add a copy button."""
        pre_blocks = list(root.findall(".//pre"))
        

        for pre in pre_blocks:

            code = pre.find("code") 
            if code is None:
                print("No <code> inside <pre>, skipping...")
                continue 

            parent = self.find_parent(root, pre)
            if parent:
                index = list(parent).index(pre)  

                parent.remove(pre)

                div = Element("div", {"class": "code-container"})

                button = Element("button", {
                    "class": "copy-btn",
                    "onclick": "copyCode(this)"
                })
                button.text = "Copy"

                pre_new = Element("pre")
                pre_new.append(code) 

                div.append(button)
                div.append(pre_new)

                parent.insert(index, div)


    def find_parent(self, root, child):
        """Finds the parent of an XML element)."""
        for parent in root.iter():
            if child in list(parent):
                return parent
        return None


class CustomMarkdownExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(CustomMarkdownProcessor(md), "custom_markdown", 15)


def convert_markdown(md_text: str) -> str:
    """Convert Markdown to HTML with custom processing."""
    extensions = [
        "extra",        
        "tables",       
        "fenced_code",  
        "attr_list",    
        "admonition",   
        "footnotes",    
        "meta",         
        CustomMarkdownExtension(),
    ]
    
    html_output = markdown.markdown(md_text, extensions=extensions)

    root = fromstring(f"<div>{html_output}</div>")

    processor = CustomMarkdownProcessor(None)
    processor.run(root)

    return tostring(root, encoding="unicode")
