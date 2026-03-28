import re

def reformat_bibliography(html):
    def replace_citation(m):
        original = m.group(0)
        p_match = re.search(r"<p>(.*?)</p>", original, re.DOTALL)
        if not p_match: return original
            
        p_text = re.sub(r"<[^>]+>", "", p_match.group(1)).strip().replace('\n', ' ')
        
        year_match = re.search(r"\((\d{4})\)", p_text)
        year = year_match.group(1) if year_match else "n.d."
        
        last_name_match = re.search(r"^([^,]+),", p_text)
        last_name = last_name_match.group(1).strip() if last_name_match else "Unknown"
        
        authors_part = p_text.split("(")[0]
        if "&" in authors_part or "et al" in authors_part or "…" in authors_part or authors_part.count(",") > 2:
            label = f"{last_name} et al., {year}"
        else:
            label = f"{last_name}, {year}"
            
        return re.sub(r'<span class="label">.*?</span></span>', f'<span class="label">{label}</span>', original, flags=re.DOTALL)

    return re.sub(r'<div class="citation"[^>]*>.*?</div>', replace_citation, html, flags=re.DOTALL)

with open("_build/html/chapter_references/zreferences.html", "r") as f:
    text = f.read()

new_text = reformat_bibliography(text)
with open("test_zref.html", "w") as f:
    f.write(new_text)
print("done")
